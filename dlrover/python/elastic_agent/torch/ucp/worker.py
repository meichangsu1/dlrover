# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0

import os
import socket
import time
import traceback

from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.comm import UcpTaskStatusUpdate
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.elastic_agent.torch.ucp.converter import UcpConverter
from dlrover.python.master.elastic_training.ucp.task_manager import (
    UcpTaskStatus,
)


def _worker_id():
    return os.getenv("DLROVER_UCP_WORKER_ID") or os.getenv(
        "POD_NAME", socket.gethostname()
    )


def main():
    worker_id = _worker_id()
    poll_interval = int(os.getenv("DLROVER_UCP_POLL_INTERVAL", "5"))
    device_type = os.getenv("DLROVER_UCP_DEVICE_TYPE", "cpu")
    os.environ.setdefault(NodeEnv.NODE_TYPE, "ucp-service")
    os.environ.setdefault(NodeEnv.NODE_ID, "0")
    client = None
    while client is None:
        client = MasterClient.singleton_instance(
            os.getenv("DLROVER_MASTER_ADDR", "")
        )
        if client is None:
            MasterClient._instance = None
            logger.info("Wait for DLRover master before starting UCP worker.")
            time.sleep(poll_interval)
    converter = UcpConverter()
    logger.info("Start DLRover UCP worker %s.", worker_id)

    while True:
        task = client.acquire_ucp_task(worker_id)
        if not task or not task.task_id:
            time.sleep(poll_interval)
            continue

        try:
            task_device = task.device_type or device_type
            ok = converter.convert(
                task.input_dir,
                task.output_dir,
                task.tmp_output_dir,
                task_device,
                task.checkpoint_dir,
                task.job_id,
                task.namespace,
                task.step,
            )
            status = (
                UcpTaskStatus.SUCCEEDED if ok else UcpTaskStatus.FAILED
            )
            err = "" if ok else "ucp converter returned false"
        except Exception:
            status = UcpTaskStatus.FAILED
            err = traceback.format_exc()
            logger.exception("UCP task failed: %s", task)

        client.update_ucp_task_status(
            UcpTaskStatusUpdate(
                task_id=task.task_id,
                status=status,
                worker_id=worker_id,
                error_message=err,
            )
        )


if __name__ == "__main__":
    main()
