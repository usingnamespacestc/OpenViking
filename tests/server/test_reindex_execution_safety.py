# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lease ownership and per-file failure isolation for reindex execution."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import openviking.service.reindex_executor as reindex_module
from openviking.server.identity import RequestContext, Role
from openviking.service.reindex_executor import ReindexExecutor, _ReindexCounters, _SourceRead
from openviking.service.task_tracker import TaskStatus
from openviking.storage.errors import LockError
from openviking.storage.queuefs.reindex_msg import ReindexMsg
from openviking.storage.queuefs.reindex_processor import ReindexProcessor
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier


def context():
    return RequestContext(user=UserIdentifier("execution_safety", "bob"), role=Role.ADMIN)


@pytest.fixture
def queued_job(monkeypatch):
    lease = {"lease_id": "accepted"}
    agfs = SimpleNamespace(
        pathlock_release=AsyncMock(),
        pathlock_acquire_tree=AsyncMock(),
        pathlock_as_borrowed=AsyncMock(return_value={"lease_ref": "borrowed"}),
    )
    fs = SimpleNamespace(
        _async_agfs=agfs,
        _uri_to_path=lambda uri, ctx: "/execution_safety/memories",
        stat=AsyncMock(return_value={"isDir": True}),
    )
    service = SimpleNamespace(
        _vlm_resolver=None,
        _vector_config_resolver=None,
        viking_fs=fs,
        vikingdb_manager=SimpleNamespace(has_queue_manager=True),
    )
    tracker = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status=TaskStatus.RUNNING)),
        start=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
        wait_for_task_work=AsyncMock(),
        build_queue_status=AsyncMock(return_value={}),
    )
    monkeypatch.setattr("openviking.server.dependencies.get_service", lambda: service)
    monkeypatch.setattr(reindex_module, "get_service", lambda: service)
    monkeypatch.setattr(reindex_module, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", lambda: tracker
    )
    monkeypatch.setattr(ReindexExecutor, "_reindex_memory", AsyncMock())
    processor = ReindexProcessor(fs)
    processor._adopt_or_reacquire = AsyncMock(return_value=lease)
    message = ReindexMsg(
        task_id="execution-task",
        uri="viking://user/bob/memories",
        object_type="memory",
        mode="vectors_only",
        account_id="execution_safety",
        user_id="bob",
        role="admin",
    )
    return SimpleNamespace(
        processor=processor,
        data={"id": "queue-1", "data": message.to_dict()},
        service=service,
        fs=fs,
        agfs=agfs,
        tracker=tracker,
        lease=lease,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "stat_timeout",
        "stat_missing",
        "fs_missing",
        "db_missing",
        "queue_missing",
        "lookup",
        "options",
    ],
)
async def test_queued_startup_failure_releases_lease_once(queued_job, monkeypatch, failure):
    job = queued_job
    if failure == "stat_timeout":
        job.fs.stat.side_effect = RuntimeError("S3 timeout")
    elif failure == "stat_missing":
        job.fs.stat.side_effect = NotFoundError("viking://user/bob/memories")
    elif failure == "fs_missing":
        job.service.viking_fs = None
    elif failure == "db_missing":
        job.service.vikingdb_manager = None
    elif failure == "queue_missing":
        job.service.vikingdb_manager.has_queue_manager = False
    elif failure == "options":
        job.data["data"].update(tags=["team=test"], tag_mode="bad")
    else:

        def unavailable():
            raise RuntimeError("Service unavailable")

        monkeypatch.setattr(reindex_module, "get_service", unavailable)
    outcome = await job.processor.on_dequeue(job.data)
    assert outcome.outcome.value == "failed"
    job.tracker.fail.assert_awaited_once()
    job.agfs.pathlock_release.assert_awaited_once_with(job.lease)
    job.agfs.pathlock_acquire_tree.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_success_reuses_and_releases_lease_once(queued_job):
    job = queued_job
    outcome = await job.processor.on_dequeue(job.data)
    assert outcome.outcome.value == "success"
    job.tracker.complete.assert_awaited_once()
    job.agfs.pathlock_as_borrowed.assert_awaited_once_with(job.lease)
    job.agfs.pathlock_release.assert_awaited_once_with(job.lease)
    job.agfs.pathlock_acquire_tree.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_cancellation_during_stat_releases_lease(queued_job):
    job = queued_job
    started = asyncio.Event()

    async def stat(*args, **kwargs):
        started.set()
        await asyncio.Future()

    job.fs.stat.side_effect = stat
    task = asyncio.create_task(job.processor.on_dequeue(job.data))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    job.agfs.pathlock_release.assert_awaited_once_with(job.lease)
    job.tracker.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_fresh_acquire_settles_lease_without_starting_work(queued_job, monkeypatch):
    job = queued_job
    started, finish = asyncio.Event(), asyncio.Event()

    async def acquire(path):
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            # Simulate a native acquisition that finishes after caller cancellation.
            await finish.wait()
        return job.lease

    job.agfs.pathlock_acquire_tree.side_effect = acquire
    monkeypatch.setattr(reindex_module, "get_task_context", lambda: None)
    task = asyncio.create_task(
        ReindexExecutor()._run(
            uri="viking://user/bob/memories",
            object_type="memory",
            mode="vectors_only",
            ctx=context(),
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    job.agfs.pathlock_release.assert_awaited_once_with(job.lease)
    job.agfs.pathlock_as_borrowed.assert_not_awaited()


@pytest.fixture
def memory_scope(monkeypatch):
    root = "viking://user/bob/memories/events"
    files = [root + "/" + name for name in ("bad.md", "healthy.md", "later.md")]
    fs = SimpleNamespace(
        exists=AsyncMock(return_value=True),
        stat=AsyncMock(return_value={"isDir": True}),
        tree=AsyncMock(return_value=[{"uri": uri, "isDir": False} for uri in files]),
    )
    db = SimpleNamespace(strict_delete=AsyncMock(side_effect=lambda ids, **kw: len(ids)))
    monkeypatch.setattr(reindex_module, "get_viking_fs", lambda: fs)
    monkeypatch.setattr(reindex_module, "get_service", lambda: SimpleNamespace(vikingdb_manager=db))
    executor = ReindexExecutor()
    executor._reindex_memory_directory_chain = AsyncMock(return_value={})
    executor._best_memory_file_summary = AsyncMock(return_value="Old summary")
    executor._upsert_context = AsyncMock()
    executor._effective_file_vectorization_concurrency = lambda: 1

    async def body(uri, **kwargs):
        text = (
            'Body\n\n<!-- MEMORY_FIELDS\n{"links":123}\n-->' if uri == files[0] else "Healthy body"
        )
        return _SourceRead(exists=True, text=text)

    executor._read_memory_body = AsyncMock(side_effect=body)
    uris = {
        "bad": files[0],
        "bad-c0": files[0] + "#chunk_0000",
        "bad-c42": files[0] + "#chunk_0042",
        "healthy": files[1],
        "healthy-c0": files[1] + "#chunk_0000",
        "healthy-stale": files[1] + "#chunk_0042",
        "later": files[2],
        "gone": root + "/gone.md",
        "gone-c0": root + "/gone.md#chunk_0000",
    }
    inventory = {key: {"id": key, "uri": uri, "level": 2} for key, uri in uris.items()}
    return SimpleNamespace(root=root, files=files, executor=executor, db=db, inventory=inventory)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_bad_memory_keeps_old_chunks_and_later_files_converge(memory_scope, concurrency):
    scope = memory_scope
    scope.executor._effective_file_vectorization_concurrency = lambda: concurrency
    counters = _ReindexCounters()
    await scope.executor._reindex_memory_vectors(
        uri=scope.root,
        ctx=context(),
        counters=counters,
        vector_inventory=scope.inventory,
    )
    assert counters.failed_records == 1
    assert counters.rebuilt_records == 2
    assert set(scope.db.strict_delete.await_args.args[0]) == {
        "bad-c0",
        "bad-c42",
        "healthy-c0",
        "healthy-stale",
        "gone",
        "gone-c0",
    }
    assert {call.kwargs["uri"] for call in scope.executor._upsert_context.await_args_list} == set(
        scope.files[1:]
    )
    assert any(scope.files[0] in warning for warning in counters.warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["read", "summary", "enqueue"])
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, LockError])
async def test_memory_lifecycle_errors_propagate_without_deletes(memory_scope, stage, error_type):
    scope = memory_scope
    error = error_type("lease lost or task cancelled")
    scope.executor._read_memory_body = AsyncMock(return_value=_SourceRead(exists=True, text="Body"))
    if stage == "read":
        if error_type is LockError:
            scope.executor._read_memory_body.return_value = _SourceRead(exists=True, error=error)
        else:
            scope.executor._read_memory_body.side_effect = error
    elif stage == "summary":
        scope.executor._best_memory_file_summary.side_effect = error
    else:
        scope.executor._upsert_context.side_effect = error
    with pytest.raises(error_type):
        await scope.executor._reindex_memory_vectors(
            uri=scope.root,
            ctx=context(),
            counters=_ReindexCounters(),
            vector_inventory=scope.inventory,
        )
    scope.db.strict_delete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, LockError])
async def test_fatal_batch_error_settles_siblings_before_return(error_type):
    started, settled = asyncio.Event(), asyncio.Event()
    tasks = []

    async def process(item):
        tasks.append(asyncio.current_task())
        if item == "bad":
            await started.wait()
            raise error_type("fatal")
        try:
            started.set()
            await asyncio.Future()
        finally:
            settled.set()

    try:
        with pytest.raises(error_type):
            await ReindexExecutor()._run_ordered_counter_batches(
                ["bad", "healthy"],
                concurrency=2,
                processor=process,
                counters=_ReindexCounters(),
            )
        assert settled.is_set(), "A sibling still owns work after the failed batch returned"
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_sibling_cleanup():
    started, cleaning, finish, settled = [asyncio.Event() for _ in range(4)]
    tasks = []

    async def process(item):
        tasks.append(asyncio.current_task())
        if item == "bad":
            await started.wait()
            raise LockError("fatal")
        try:
            started.set()
            await asyncio.Future()
        finally:
            cleaning.set()
            await finish.wait()
            settled.set()

    operation = asyncio.create_task(
        ReindexExecutor()._run_ordered_counter_batches(
            ["bad", "healthy"],
            concurrency=2,
            processor=process,
            counters=_ReindexCounters(),
        )
    )
    try:
        await asyncio.wait_for(cleaning.wait(), 1)
        operation.cancel()
        await asyncio.sleep(0)
        operation.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 0.01)
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert settled.is_set()
    finally:
        finish.set()
        await asyncio.gather(operation, return_exceptions=True)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_external_cancellation_waits_for_all_file_cleanup():
    started, cleaning = asyncio.Queue(), asyncio.Queue()
    finish = asyncio.Event()
    settled, tasks = set(), []

    async def process(item):
        tasks.append(asyncio.current_task())
        started.put_nowait(item)
        try:
            await asyncio.Future()
        finally:
            cleaning.put_nowait(item)
            await finish.wait()
            settled.add(item)

    operation = asyncio.create_task(
        ReindexExecutor()._run_ordered_counter_batches(
            ["one", "two"],
            concurrency=2,
            processor=process,
            counters=_ReindexCounters(),
        )
    )
    try:
        for _ in range(2):
            await asyncio.wait_for(started.get(), 1)
        operation.cancel()
        for _ in range(2):
            await asyncio.wait_for(cleaning.get(), 1)
        operation.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 0.01)
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert settled == {"one", "two"}
    finally:
        finish.set()
        await asyncio.gather(operation, return_exceptions=True)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
