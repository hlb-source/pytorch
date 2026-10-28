# Owner(s): ["module: inductor"]

"""NPU 冒烟版 foreach 水平融合测试（对应 test_foreach.py::test_single_list 的最小子集）。

目的：验证在 NPU（torch_npu + 其注册的 Inductor 后端，通常配合 triton-ascend）上，
一条 foreach 调用（列表内张量 shape 不同）会被调度器水平融合为恰好 1 个 kernel。

运行环境要求（任一不满足会 skip 并给出原因）：
  1. torch_npu 可用且 torch.npu.is_available() 为 True；
  2. torch_npu 注册了 Inductor 的 npu 后端（get_scheduling_for_device("npu") 非 None）。

用法：在 NPU 机器上 `python test/inductor/test_foreach_npu.py`（或 pytest 单文件）。
若并行编译子进程出问题，可先 export TORCHINDUCTOR_COMPILE_THREADS=1 再试。
"""

import unittest

import torch

try:
    import torch_npu  # noqa: F401  — import 副作用：注册 npu 设备与 Inductor 后端

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False

from torch._inductor import metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code

NPU = "npu"


def _inductor_npu_backend_registered() -> bool:
    from torch._inductor.codegen.common import get_scheduling_for_device
    from torch._inductor.codegen.common import init_backend_registration

    init_backend_registration()
    return get_scheduling_for_device(NPU) is not None


@unittest.skipIf(not HAS_NPU, "torch_npu not available")
class ForeachNpuSmokeTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _inductor_npu_backend_registered():
            raise unittest.SkipTest(
                "Inductor npu backend not registered; check torch_npu version "
                "(needs a release that registers torch_npu.inductor)"
            )

    def _check_single_list_fused(self, eager_fn, args, expected_kernels=1):
        torch.manual_seed(42)
        expected = eager_fn(*args)
        metrics.reset()
        torch._dynamo.reset()
        torch.manual_seed(42)
        actual, codes = run_and_get_code(torch.compile(eager_fn), *args)
        torch.testing.assert_close(actual, expected)
        kernel_names = [
            ln for code in codes if isinstance(code, str)
            for ln in code.splitlines() if ln.startswith("def ")
        ]
        self.assertEqual(
            metrics.generated_kernel_count,
            expected_kernels,
            msg=f"expected {expected_kernels} kernel(s), got {metrics.generated_kernel_count}; kernels: {kernel_names}",
        )
        return codes

    def test_foreach_unary_single_list(self):
        """一元 foreach：op([a0, a1])，不同 shape，必须融合为 1 个 kernel。"""
        a0 = torch.rand(10, 10, device=NPU)
        a1 = torch.rand(20, 20, device=NPU)
        for op in (torch._foreach_neg, torch._foreach_sqrt):
            with self.subTest(op=op.__name__):
                self._check_single_list_fused(lambda a, b: op([a, b]), (a0, a1))

    def test_foreach_binary_add_single_list(self):
        """二元 foreach：_foreach_add([a0, a1], [b0, b1])，必须融合为 1 个 kernel。"""
        a0 = torch.rand(10, 10, device=NPU)
        a1 = torch.rand(20, 20, device=NPU)
        b0 = torch.rand(10, 10, device=NPU)
        b1 = torch.rand(20, 20, device=NPU)

        def fn(a0, a1, b0, b1):
            return torch._foreach_add([a0, a1], [b0, b1])

        self._check_single_list_fused(fn, (a0, a1, b0, b1))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_NPU:
        run_tests()
