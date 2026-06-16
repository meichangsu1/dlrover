# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0

import os
import threading
import time
import uuid
from typing import Dict, Optional

from dlrover.python.common import comm
from dlrover.python.common.log import default_logger as logger


class UcpTaskStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class UcpTaskManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, comm.UcpTask] = {}
        self._latest_ready: Dict[str, comm.UcpTask] = {}
        self._max_retries = int(os.getenv("DLROVER_UCP_MAX_RETRIES", "3"))
        self._lease_timeout = int(
            os.getenv("DLROVER_UCP_WORKER_LEASE_TIMEOUT", "300")
        )

    @classmethod
    def singleton_instance(cls):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = UcpTaskManager()
        return cls._instance

    @staticmethod
    def task_key(namespace: str, job_id: str, step: int):
        return f"{namespace}:{job_id}:{step}"

    def report_checkpoint_ready(
        self, request: comm.ReportCheckpointReady
    ) -> comm.UcpTask:
        key = self.task_key(request.namespace, request.job_id, request.step)
        now = int(time.time())
        with self._lock:
            task = self._tasks.get(key)
            if task:
                logger.info("UCP task already exists: %s", task)
                return task

            task_id = request.task_id or str(uuid.uuid4())
            output_dir = request.output_dir
            tmp_output_dir = (
                request.tmp_output_dir
                or f"{output_dir}_tmp_{task_id[:8]}"
            )
            task = comm.UcpTask(
                task_id=task_id,
                job_id=request.job_id,
                namespace=request.namespace,
                step=request.step,
                checkpoint_dir=request.checkpoint_dir,
                input_dir=request.input_dir,
                output_dir=output_dir,
                tmp_output_dir=tmp_output_dir,
                framework=request.framework,
                backend=request.backend,
                device_type=request.device_type,
                max_retries=request.max_retries or self._max_retries,
                timeout_seconds=request.timeout_seconds,
                status=UcpTaskStatus.PENDING,
                created_at=now,
                updated_at=now,
            )
            self._tasks[key] = task
            logger.info("Create UCP task: %s", task)
            return task

    def acquire_task(self, worker_id: str) -> comm.UcpTask:
        now = int(time.time())
        with self._lock:
            for task in self._tasks.values():
                if task.status == UcpTaskStatus.RUNNING:
                    if now - task.updated_at <= self._lease_timeout:
                        continue
                    logger.warning("Requeue expired UCP task %s", task.task_id)
                    task.status = UcpTaskStatus.PENDING

                if task.status != UcpTaskStatus.PENDING:
                    continue

                task.status = UcpTaskStatus.RUNNING
                task.worker_id = worker_id
                task.started_at = now if task.started_at == 0 else task.started_at
                task.updated_at = now
                return task
        return comm.UcpTask()

    def update_status(self, update: comm.UcpTaskStatusUpdate) -> bool:
        with self._lock:
            task = self._find_task(update.task_id)
            if not task:
                logger.warning("Unknown UCP task status update: %s", update)
                return False

            task.worker_id = update.worker_id or task.worker_id
            task.error_message = update.error_message
            task.updated_at = int(time.time())

            if update.status == UcpTaskStatus.SUCCEEDED:
                task.status = UcpTaskStatus.SUCCEEDED
                task.finished_at = task.updated_at
                latest_key = self._latest_key(task.namespace, task.job_id)
                self._latest_ready[latest_key] = task
                logger.info("UCP task succeeded: %s", task)
                return True

            if update.status == UcpTaskStatus.FAILED:
                task.retry_count += 1
                if task.retry_count < task.max_retries:
                    task.status = UcpTaskStatus.PENDING
                    task.worker_id = ""
                    logger.warning(
                        "Retry UCP task %s: %s/%s",
                        task.task_id,
                        task.retry_count,
                        task.max_retries,
                    )
                else:
                    task.status = UcpTaskStatus.FAILED
                    task.finished_at = task.updated_at
                    logger.error("UCP task failed: %s", task)
                return True

            task.status = update.status
            return True

    def get_task(self, namespace: str, job_id: str, step: int) -> comm.UcpTask:
        with self._lock:
            return self._tasks.get(
                self.task_key(namespace, job_id, step), comm.UcpTask()
            )

    def latest_task(self) -> Optional[comm.UcpTask]:
        with self._lock:
            if not self._tasks:
                return None
            return max(self._tasks.values(), key=lambda t: t.step)

    def get_resume_checkpoint(self, namespace: str, job_id: str) -> str:
        with self._lock:
            task = self._latest_ready.get(self._latest_key(namespace, job_id))
            if not task:
                return ""
            checkpoint_root = os.path.dirname(task.output_dir)
            return checkpoint_root or task.output_dir

    def _find_task(self, task_id: str) -> Optional[comm.UcpTask]:
        for task in self._tasks.values():
            if task.task_id == task_id:
                return task
        return None

    @staticmethod
    def _latest_key(namespace: str, job_id: str):
        return f"{namespace}:{job_id}"
