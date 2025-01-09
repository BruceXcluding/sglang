import glob
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()


def is_cuda() -> bool:
    """Return whether it is CUDA on the NVIDIA CUDA platform."""
    return torch.cuda.is_available() and torch.version.cuda


def is_hip() -> bool:
    """Return whether it is HIP on the AMD ROCm platform."""
    return torch.cuda.is_available() and torch.version.hip


def get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


def update_wheel_platform_tag():
    wheel_dir = Path("dist")
    old_wheel = next(wheel_dir.glob("*.whl"))
    new_wheel = wheel_dir / old_wheel.name.replace(
        "linux_x86_64", "manylinux2014_x86_64"
    )
    old_wheel.rename(new_wheel)


cutlass = root / "3rdparty" / "cutlass"
include_dirs = [
    cutlass.resolve() / "include",
    cutlass.resolve() / "tools" / "util" / "include",
]
nvcc_flags = [
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

ck = root / "3rdparty" / "composable_kernel"
ck_include_dirs = [
    ck.resolve() / "include",
    root / "src" / "sgl-kernel-amd" / "csrc",
]

hipcc_flags = [
    "-D__HIP_PLATFORM_AMD__=1",
    "--amdgpu-target=gfx90a,gfx940,gfx941,gfx942",
]

if is_cuda():
    cxx_flags = ["-O3"]
    libraries = ["c10", "torch", "torch_python"]
    extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib"]
    ext_modules = [
        CUDAExtension(
            name="sgl_kernel.ops._kernels",
            sources=[
                "src/sgl-kernel/csrc/trt_reduce_internal.cu",
                "src/sgl-kernel/csrc/trt_reduce_kernel.cu",
                "src/sgl-kernel/csrc/moe_align_kernel.cu",
                "src/sgl-kernel/csrc/sgl_kernel_ops.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "nvcc": nvcc_flags,
                "cxx": cxx_flags,
            },
            libraries=libraries,
            extra_link_args=extra_link_args,
        ),
    ]

    setup(
        name="sgl-kernel",
        version=get_version(),
        packages=["sgl_kernel", "sgl_kernel.ops", "sgl_kernel.csrc"],
        package_dir={"sgl_kernel": "src/sgl-kernel"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        install_requires=["torch"],
    )
elif is_hip():
    ext_modules = [
        CUDAExtension(
            name="sgl_kernel.ops._kernels",
            sources=[
                "src/sgl-kernel/csrc/moe_align_kernel.cu",
                "src/sgl-kernel-amd/csrc/ck/gemm_a8w8/gemm_a8w8_subblock.cu",
                "src/sgl-kernel-amd/csrc/sgl_kernel_amd_ops.cu",
            ],
            # ]+ glob.glob("src/sgl-kernel-amd/csrc/ck/gemm_a8w8/impl/*.cu"), #TODO GEMM TUNING IMPL
            include_dirs=ck_include_dirs,
            extra_compile_args={
                "nvcc": hipcc_flags
                + [
                    "-O3",
                    "-fPIC",
                ],
                "cxx": ["-O3"],
            },
            libraries=["hiprtc", "amdhip64", "c10", "torch", "torch_python"],
            extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        ),
    ]

    setup(
        name="sgl-kernel",
        version=get_version(),
        packages=["sgl_kernel", "sgl_kernel.ops", "sgl_kernel.csrc"],
        package_dir={"sgl_kernel": "src/sgl-kernel-amd"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        install_requires=["torch"],
    )
else:
    raise RuntimeError(
        "Neither CUDA nor ROCm environment detected. Set either CUDA_HOME or ROCM_PATH"
    )


update_wheel_platform_tag()
