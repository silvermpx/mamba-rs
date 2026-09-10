#!/usr/bin/env bash
set -euo pipefail
evidence_dir=/root/sm120-lower-toolkits-20260910
mkdir -p "$evidence_dir"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$evidence_dir/device-before.csv"
readlink -f /usr/local/cuda > "$evidence_dir/default-before.txt"
packages=(
  cuda-nvcc-12-8=12.8.93-1 cuda-nvrtc-dev-12-8=12.8.93-1
  cuda-cudart-dev-12-8=12.8.90-1 cuda-cccl-12-8=12.8.90-1
  libcublas-dev-12-8=12.8.5.5-1
  cuda-nvcc-13-0=13.0.88-1 cuda-nvrtc-dev-13-0=13.0.88-1
  cuda-cudart-dev-13-0=13.0.96-1 cuda-cccl-13-0=13.0.85-1
  libcublas-dev-13-0=13.1.1.3-1
)
apt-get -s --no-install-recommends install "${packages[@]}" > "$evidence_dir/simulation.log"
grep -F '0 upgraded, 27 newly installed, 0 to remove' "$evidence_dir/simulation.log"
if grep -E '^Inst (cuda-drivers|nvidia-driver|linux-image|linux-modules)' "$evidence_dir/simulation.log"; then
    printf 'Unexpected driver/kernel change; refusing install\n' >&2
    exit 1
fi
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}" > "$evidence_dir/install.log" 2>&1
for toolkit in 12.8 13.0 13.2; do
    "/usr/local/cuda-$toolkit/bin/nvcc" --version > "$evidence_dir/nvcc-$toolkit.txt"
done
dpkg-query -W 'cuda-nvcc-*' 'cuda-nvrtc-*' 'cuda-cudart-*' 'cuda-cccl-*' 'libcublas-*' > "$evidence_dir/packages.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$evidence_dir/device-after.csv"
readlink -f /usr/local/cuda > "$evidence_dir/default-after.txt"
cmp "$evidence_dir/device-before.csv" "$evidence_dir/device-after.csv"
cmp "$evidence_dir/default-before.txt" "$evidence_dir/default-after.txt"
printf 'CUDA12.8/13.0 installed; driver and default unchanged\n'
