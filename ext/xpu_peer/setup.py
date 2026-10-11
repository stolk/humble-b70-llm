# Build with the oneAPI compiler on PATH (source /opt/intel/oneapi/setvars.sh):
#   pip install --no-build-isolation --no-deps ext/xpu_peer
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, SyclExtension

setup(
    name="xpu_peer",
    version="0.1.0",
    packages=["xpu_peer"],
    ext_modules=[
        SyclExtension(
            "xpu_peer._C",
            ["csrc/xpu_peer.cpp"],
            libraries=["ze_loader"],
            # Host-side SYCL only (no kernels), so no -fsycl; the deprecations
            # are inside the SYCL headers.
            extra_compile_args={"cxx": ["-O2", "-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING",
                                        "-Wno-deprecated-declarations"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
