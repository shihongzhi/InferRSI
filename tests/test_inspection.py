"""Inspection facts must not become accelerator compatibility claims."""

import json
import sys
import unittest
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from inferrsi.inspection import inspect_environment


class InspectionTests(unittest.TestCase):
    def inspect_cuda(self, rows, ids=("0",), model=None):
        def command(argv):
            if "-c" in argv:
                stdout = json.dumps({"python": "test", "packages": {"torch": "test-version"}})
            elif "topo" in argv:
                stdout = "test topology"
            else:
                stdout = rows
            return {"argv": argv, "returncode": 0, "stdout": stdout, "stderr": ""}

        with (
            patch(
                "inferrsi.execution.repository_identity",
                return_value={"available": True, "head": "test"},
            ),
            patch("inferrsi.inspection._command", side_effect=command),
        ):
            return inspect_environment(".", "vllm_omni", "cuda", device_ids=ids, device_model=model)

    def test_requested_device_and_model_are_checked(self):
        rows = "0, GPU-A, NVIDIA A100, 999.0\n1, GPU-B, NVIDIA H200, 999.0\n"
        result = self.inspect_cuda(rows, ids=("1",), model="H200")
        self.assertEqual(result["device"]["status"], "available")
        self.assertEqual(result["device"]["devices"][1]["driver_version"], "999.0")
        self.assertEqual(result["compatibility"], "unverified")
        mismatch = self.inspect_cuda(rows, ids=("0",), model="H200")
        self.assertEqual(mismatch["device"]["status"], "device_model_mismatch")

    def test_successful_tool_without_requested_gpu_is_not_available(self):
        empty = self.inspect_cuda("")
        self.assertEqual(empty["device"]["status"], "unavailable")
        missing = self.inspect_cuda("0, GPU-A, NVIDIA H200, 999.0\n", ids=("8",))
        self.assertFalse(missing["device"]["selection_verified"])
        self.assertEqual(missing["device"]["status"], "unavailable")


class MlxInspectionTests(unittest.TestCase):
    def inspect_mlx(self, chip, model=None):
        def command(argv):
            if "-c" in argv:
                stdout = json.dumps({"python": "test", "packages": {"mlx": None}})
            elif argv[0] == "sysctl":
                stdout = chip
            else:
                stdout = ""
            return {"argv": argv, "returncode": 0, "stdout": stdout, "stderr": ""}

        with (
            patch(
                "inferrsi.execution.repository_identity",
                return_value={"available": True, "head": "test"},
            ),
            patch("inferrsi.inspection._command", side_effect=command),
        ):
            return inspect_environment(".", "vllm_omni", "mlx", device_model=model)

    def test_apple_silicon_chip_is_available(self):
        result = self.inspect_mlx("Apple M3 Max\n")
        self.assertEqual(result["device"]["status"], "available")
        self.assertTrue(result["device"]["selection_verified"])
        self.assertEqual(result["device"]["chip"], "Apple M3 Max")
        self.assertEqual(result["backend"], "mlx")
        self.assertEqual(result["compatibility"], "unverified")

    def test_non_apple_chip_is_not_available(self):
        result = self.inspect_mlx("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz\n")
        self.assertEqual(result["device"]["status"], "unavailable")
        self.assertFalse(result["device"]["selection_verified"])

    def test_requested_chip_model_must_match(self):
        mismatch = self.inspect_mlx("Apple M3 Max\n", model="M2 Ultra")
        self.assertEqual(mismatch["device"]["status"], "device_model_mismatch")
        matched = self.inspect_mlx("Apple M3 Max\n", model="m3 max")
        self.assertEqual(matched["device"]["status"], "available")


class PythonInspectionTests(unittest.TestCase):
    def test_real_python_reports_requested_versions(self):
        result = inspect_environment(
            str(Path(__file__).resolve().parents[1]),
            "vllm_omni",
            "cpu",
            python=sys.executable,
        )
        probe = result["target_python"]
        self.assertEqual(probe["returncode"], 0, probe["stderr"])
        self.assertNotIn("parse_error", probe)
        environment = probe["environment"]
        self.assertEqual(environment["python"], sys.version)
        names = (
            "torch",
            "vllm",
            "vllm-omni",
            "vllm-afd-plugin",
            "vllm-rlt",
            "torch-npu",
            "mlx",
            "nvidia-nccl-cu12",
            "nvidia-nccl-cu13",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cuda-runtime-cu13",
        )
        self.assertEqual(set(environment["packages"]), set(names))
        for name in names:
            with self.subTest(package=name):
                try:
                    expected = metadata.version(name)
                except metadata.PackageNotFoundError:
                    expected = None
                self.assertEqual(environment["packages"][name], expected)
        self.assertEqual(result["device"]["status"], "available")
        self.assertEqual(result["compatibility"], "unverified")


if __name__ == "__main__":
    unittest.main()
