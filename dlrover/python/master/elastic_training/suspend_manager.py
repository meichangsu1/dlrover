# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0

import threading
from dataclasses import dataclass
from typing import Set

from dlrover.python.common import comm
from dlrover.python.common.log import default_logger as logger


class SuspendState:
    RUNNING = "RUNNING"
    SUSPENDING = "SUSPENDING"
    SUSPENDED_ZERO_WORKER = "SUSPENDED_ZERO_WORKER"
    RESUMING = "RESUMING"


@dataclass
class SuspendContext:
    state: str = SuspendState.RUNNING
    reason: str = ""
    step: int = 0
    task_id: str = ""


class SuspendManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._ctx = SuspendContext()
        self._ready_nodes: Set[int] = set()
        self._job_manager = None

    @classmethod
    def singleton_instance(cls):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = SuspendManager()
        return cls._instance

    def set_job_manager(self, job_manager):
        self._job_manager = job_manager

    def request_suspend_to_zero(self, reason: str = ""):
        with self._lock:
            if self._ctx.state in (
                SuspendState.SUSPENDING,
                SuspendState.SUSPENDED_ZERO_WORKER,
            ):
                return
            logger.info("Request suspend-to-zero: %s", reason)
            self._ctx = SuspendContext(
                state=SuspendState.SUSPENDING, reason=reason
            )
            self._ready_nodes.clear()

    def request_resume(self):
        with self._lock:
            if self._ctx.state == SuspendState.RUNNING:
                return
            logger.info("Request resume from suspend-to-zero.")
            self._ctx = SuspendContext()
            self._ready_nodes.clear()

    def get_status(self) -> comm.SuspendStatus:
        with self._lock:
            return comm.SuspendStatus(
                state=self._ctx.state, reason=self._ctx.reason
            )

    def report_ready(self, ready: comm.SuspendReady) -> bool:
        with self._lock:
            if self._ctx.state != SuspendState.SUSPENDING:
                logger.info(
                    "Ignore suspend ready from node %s when state is %s.",
                    ready.node_id,
                    self._ctx.state,
                )
                return True
            self._ready_nodes.add(ready.node_id)
            if ready.step:
                self._ctx.step = ready.step
            if ready.task_id:
                self._ctx.task_id = ready.task_id
            logger.info(
                "Node %s is ready to suspend: step=%s, task_id=%s.",
                ready.node_id,
                ready.step,
                ready.task_id,
            )

        self._scale_workers_to_zero()
        return True

    def _scale_workers_to_zero(self):
        if not self._job_manager:
            logger.warning("Skip scaling workers to zero for no job manager.")
            return

        try:
            if hasattr(self._job_manager, "suspend_training_workers"):
                self._job_manager.suspend_training_workers()
            else:
                self._job_manager.remove_training_nodes()
            with self._lock:
                self._ctx.state = SuspendState.SUSPENDED_ZERO_WORKER
            logger.info("Workers have been requested to scale to zero.")
        except Exception:
            logger.exception("Failed to scale workers to zero.")


def get_suspend_manager() -> SuspendManager:
    return SuspendManager.singleton_instance()
