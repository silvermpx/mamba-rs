# Ada CUDA13.0 side-by-side installation

2026-09-06. Owner had explicitly authorized installing missing toolkits on
this Ada host. Main installed the versioned NVIDIA toolkit after the prior
AUTO timing lane was released; no new timing was permitted during installation.

Reviewed simulation:59 new packages,0 upgrades,0 removals. Free disk167GiB.
Command:

```sh
env DEBIAN_FRONTEND=noninteractive apt-get -y --no-remove --no-upgrade \
  --no-install-recommends install cuda-toolkit-13-0=13.0.3-1
```

This follows NVIDIA's documented version-pinned/side-by-side toolkit approach:
[NVIDIA CUDA13.0 installation guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-linux/index.html#package-upgrades).
No `cuda`/driver metapackage, upgrade, autoremove or explicit service restart
was requested. `cuda-driver-dev-13-0` is the toolkit development package, not
a new installed GPU driver.

Post-install probes exited0 and confirm install-ok status for toolkit13.0.3-1,
NVRTC/NVCC13.0.88 and cuBLAS component13.1.1.3. NVCC reports CUDA13.0;
`/usr/local/cuda` still resolves to13.2; GPU driver remains595.45.04.
APT repaired the CUDA alternatives link registration while retaining13.2 and
registered packaged Nsight command alternatives. Its service-restart notices
were deferred; main did not act on its process/session list or send signals.

The captured installation transcript is complete through the package and
needrestart summary; CRLF is normalized to LF in `install.log`. Its tool
wrapper failed when storing an absent completed-session identifier, so no
APT exit code is asserted from that wrapper. Installation acceptance is based
on the subsequent successful direct package/NVCC/driver/default-path probes
in `post-install.log`, not on the wrapper error or mere directory presence.

The census worker may now time only after its own matching-version build and
functional checks finish and idle preflight passes. CUDA13.0 performance and
numeric qualification remain separate from installing the toolkit.
