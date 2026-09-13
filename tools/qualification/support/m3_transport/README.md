# Mamba-3 transport qualification

This manual instrument checks the functions selected by `Mamba3Kernels`, not
separately compiled candidate entry points. It requires an idle Ada device,
native `sm_89`, and NVRTC 12.8, 13.0, or 13.2.

```sh
cargo test --release --features cuda,qualification \
  --test m3_transport_qualification -- --ignored --nocapture --test-threads=1
```

Run it once per toolkit. Each process compiles capacities 16, 32, and 64 and
batches its cases under each compiled registry. Toolkit environment variables
must point to matching headers and NVRTC libraries. Check GPU utilization and
other compute processes before running; do not overlap with model workloads.

`legacy_chunked.cu` is the exact pre-transport fragment from commit `d91e7641`.
Its SHA-256 is asserted before compilation. The oracle combines that fragment
with the same six unchanged fragments used by the production module. Source
and canonical PTX digests are printed with the compiler version and capacity.
This isolates the transport change; it is not a frozen oracle for subsequent
changes to those other six files or a substitute for release-wide bit ledgers.

The test checks selector identities and launch geometry, then compares all six
dQKV outputs, all five dQKTheta outputs, both angle outputs, and axis-0 results.
Every input and output has independent guards. Inputs must remain unchanged.
Each eager comparison is followed by two one-launch graph replays with outputs
restored before each replay. Accumulating reducers restore identical finite
initial values; overwrite kernels use different poisons in the two arms.
The matrix includes admitted shapes, neighboring fallback shapes, partial
chunks, direct/staged dQKTheta, and reduction cancellation, signed zero,
subnormal and mixed-exponent fixtures.

The instrument deliberately fails when `MAMBA_RS_TRANSPORT_NEGATIVE=eager`
omits the candidate eager launch or `MAMBA_RS_TRANSPORT_NEGATIVE=graph` captures
work only into the oracle buffers. These modes check that stale output cannot
satisfy the comparison. Run each negative mode separately and require a raw
output mismatch, not just a compilation or capture failure.

There are no timing assertions. Kernel screens and route-stamped model
benchmarks establish speed; this test checks exact output and dispatch wiring.
