# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0

import importlib.util
import json
import os
import shutil
import sys
import time
from typing import Optional

from dlrover.python.common.log import default_logger as logger


class UcpConverter:
    def get_deepspeed_install_dir(self):
        spec = importlib.util.find_spec("deepspeed")
        if spec and spec.origin:
            return os.path.dirname(spec.origin)
        return ""

    def convert(
        self,
        input_dir: str,
        output_dir: str,
        tmp_output_dir: Optional[str] = None,
        device_type: str = "cpu",
        checkpoint_dir: str = "",
        job_id: str = "",
        namespace: str = "",
        step: int = 0,
    ) -> bool:
        from packaging import version
        import torch
        from torch.distributed.elastic.multiprocessing.api import (
            SubprocessHandler,
        )

        def version_less_than_230():
            current_version = version.parse(torch.__version__).base_version
            return version.parse(current_version) <= version.parse("2.2.2")

        tmp_output_dir = tmp_output_dir or output_dir
        if os.path.exists(tmp_output_dir):
            shutil.rmtree(tmp_output_dir)

        cmd = os.getenv("PYTHON_EXEC", sys.executable)
        deepspeed_dir = self.get_deepspeed_install_dir()
        script = os.path.join(
            deepspeed_dir, "checkpoint", "ds_to_universal.py"
        )
        args_list = [
            script,
            "--input_folder",
            input_dir,
            "--output_folder",
            tmp_output_dir,
            "--inject_missing_state",
        ]
        if device_type != "cpu":
            args_list.extend(["--device", device_type])

        logger.info("Run UCP converter: %s %s", cmd, args_list)
        if version_less_than_230():
            handler = SubprocessHandler(cmd, tuple(args_list), {}, "", "")
        else:
            handler = SubprocessHandler(cmd, tuple(args_list), {}, "", "", 0)
        ret = handler.proc.wait()
        if ret != 0:
            logger.error("UCP converter returned non-zero exit code %s", ret)
            return False

        self._commit_output(
            tmp_output_dir,
            output_dir,
            input_dir,
            checkpoint_dir,
            job_id,
            namespace,
            step,
        )
        return True

    def _commit_output(
        self,
        tmp_output_dir,
        output_dir,
        input_dir,
        checkpoint_dir,
        job_id,
        namespace,
        step,
    ):
        if tmp_output_dir != output_dir:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.rename(tmp_output_dir, output_dir)

        os.makedirs(output_dir, exist_ok=True)
        metadata = {
            "job_id": job_id,
            "namespace": namespace,
            "step": step,
            "source_checkpoint": input_dir,
            "universal_checkpoint": output_dir,
            "created_by": "dlrover-ucp-service",
            "created_at": int(time.time()),
            "status": "SUCCEEDED",
        }
        with open(
            os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f)
        with open(os.path.join(output_dir, "_SUCCESS"), "w") as f:
            f.write("")
        if checkpoint_dir:
            with open(
                os.path.join(checkpoint_dir, "ucp.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(str(step))
