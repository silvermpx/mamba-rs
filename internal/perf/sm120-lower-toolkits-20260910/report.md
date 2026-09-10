# RTX 5090 lower-toolkit setup

2026-09-10, current rental `61.32.91.194:18481`, Ubuntu24.04. Before this
setup only CUDA13.2 was installed. This is environment preparation, not kernel
qualification or a performance result.

Installed the CUDA12.8 and13.0 compiler, NVRTC development/runtime, CUDA runtime
headers, CCCL and cuBLAS development/runtime packages from the configured
official NVIDIA repository. Explicit component versions match the Ada host.
The exact package list is in `packages.txt`; `install-sm120-lower-toolkits.sh`
contains the pinned request and checks. The package-manager simulation confirmed
0 upgrades,27 new packages,0 removals and no GPU-driver/kernel packages.

The install downloaded2190MB and added5367MB. Both compilers report the expected
versions: NVCC12.8.93 and13.0.88. Existing CUDA13.2 remains installed.
`device-before.csv` and `device-after.csv` compare equal; driver595.84 and GPU
identity are unchanged. `default-before.txt` and `default-after.txt` also compare
equal: `/usr/local/cuda` still resolves to `/usr/local/cuda-13.2`.

Side-by-side versioned CUDA packages follow NVIDIA's
[Linux installation guidance](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/).
No driver upgrade, reboot, model unload or GPU benchmark was performed by this
setup. Subsequent builds must set CUDA_HOME/PATH/LD_LIBRARY_PATH explicitly and
keep each toolkit's artifact/runtime receipts separate. Merely installing a
toolkit does not admit any kernel to AUTO.
