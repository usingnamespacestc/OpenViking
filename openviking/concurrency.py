# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Local synchronization for async callers running on different threads."""

import asyncio
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import Future
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def bounded_map(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
) -> list[R]:
    """Apply ``worker`` with a fixed number of tasks and preserve input order.

    A failed worker prevents new items from being claimed, then all active
    workers are joined before the original failure is re-raised.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    iterator = iter(enumerate(items))
    cursor_lock = asyncio.Lock()
    stop = asyncio.Event()
    results: dict[int, R] = {}

    async def next_item() -> tuple[int, T] | None:
        async with cursor_lock:
            if stop.is_set():
                return None
            try:
                return next(iterator)
            except StopIteration:
                return None

    async def run_worker() -> None:
        while item := await next_item():
            index, value = item
            try:
                results[index] = await worker(value)
            except BaseException:
                stop.set()
                raise

    workers = [asyncio.create_task(run_worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        stop.set()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    return [results[index] for index in range(len(results))]


class AsyncSemaphore:
    """Limit concurrent operations without binding them to one event loop.

    The thread lock only protects admission and handoff. Contended callers
    await a future on their own loop; no executor thread is occupied waiting.

    ``priority`` callers are woken before ordinary waiters, but after
    ``max_priority_burst`` consecutive grants that bypassed an eligible ordinary
    waiter, the next permit goes to the ordinary queue so it cannot starve.
    ``reserved`` permits are additionally usable only by ``priority`` callers.
    """

    def __init__(self, value: int = 1, *, reserved: int = 0, max_priority_burst: int = 4) -> None:
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        if reserved < 0 or (reserved and reserved >= value):
            raise ValueError("Semaphore reserved value must be >= 0 and < initial value")
        if max_priority_burst < 1:
            raise ValueError("Semaphore max_priority_burst must be >= 1")
        self._value = value
        self._reserved = reserved
        self._ordinary_limit = value - reserved
        self._ordinary_held = 0
        self._max_priority_burst = max_priority_burst
        self._priority_streak = 0
        self._lock = threading.Lock()
        self._waiters: deque[Future[None]] = deque()
        self._priority_waiters: deque[Future[None]] = deque()

    async def acquire(self, *, priority: bool = False) -> None:
        waiters = self._priority_waiters if priority else self._waiters
        with self._lock:
            if not waiters and self._can_take(priority):
                self._take(priority)
                return
            waiter: Future[None] = Future()
            waiters.append(waiter)
        try:
            await asyncio.wrap_future(waiter)
        except BaseException:
            with self._lock:
                try:
                    waiters.remove(waiter)
                except ValueError:
                    # A granted slot must be returned even if cancellation
                    # reached this loop before the grant callback did.
                    granted = not waiter.cancelled()
                else:
                    granted = False
            if granted:
                self.release(priority=priority)
            raise

    def release(self, *, priority: bool = False) -> None:
        with self._lock:
            self._value += 1
            if self._reserved and not priority:
                self._ordinary_held -= 1
            self._wake()

    def _can_take(self, priority: bool) -> bool:
        if priority or not self._reserved:
            return self._value > 0
        return self._value > 0 and self._ordinary_held < self._ordinary_limit

    def _take(self, priority: bool) -> None:
        self._value -= 1
        if self._reserved and not priority:
            self._ordinary_held += 1

    def _wake(self) -> None:
        while True:
            ordinary_ready = bool(self._waiters) and self._can_take(False)
            priority_ready = bool(self._priority_waiters) and self._can_take(True)
            if priority_ready and (
                not ordinary_ready or self._priority_streak < self._max_priority_burst
            ):
                priority, waiters = True, self._priority_waiters
            elif ordinary_ready:
                priority, waiters = False, self._waiters
            else:
                return
            waiter = waiters.popleft()
            if not waiter.set_running_or_notify_cancel():
                continue
            self._take(priority)
            # Only grants that passed over an eligible ordinary waiter count.
            self._priority_streak = self._priority_streak + 1 if priority and ordinary_ready else 0
            waiter.set_result(None)

    async def __aenter__(self) -> "AsyncSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()
