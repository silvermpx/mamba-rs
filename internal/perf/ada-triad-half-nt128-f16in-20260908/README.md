# Loaded NT128 and aligned F16 NN d768-in: no additional Fast winner

Ada / CUDA13.2, 2026-09-08. One focused typed build, two exact processes.

| Candidate / dtype | Paired candidate/current p50 | Paired candidate/Fast p50 | Result |
| --- | ---: | ---: | --- |
| NN S3 d768-in F16 | .6775–.6808 | 1.0210–1.0400 | Useful current improvement; Fast gap2–4% |
| NT128 d768-out F16 | 1.0630–1.0637 | 1.4392–1.4951 | Stop; slower than current |
| NT128 d768-out BF16 | 1.0619–1.0657 | 1.4830–1.5154 | Stop; slower than current |

NN current is forced TC128; NT current is forced TC64, not new public AUTO
qualification. Keep the NN current-speed benefit in the shortlist, but do not
count it as a Fast win. The existing five aligned S3 Fast-winning cells are
unchanged. NT128 is already loaded and was previously unmeasured on this Ada
shape; its loss does not authorize deleting the route for other cells/devices.

Both exact tests PASS with 256B logical alignment, resource/guard/repeat
eager+graph current-bit checks and separate Fast own-bit checks. NT128:
grid192/block256/dynamic73728; no new NVRTC kernel. Once7, ABBA/BAAB,
20 GEMMs/observation. Root replayed all168 brackets/672 observations and
quantiles. Cache stable, exact-list and quiet/drain checks pass.

Exact tests (bare names in this binary):

- `ada_half_nt_d768_out_loaded_tc128_vs_current_and_fast_discovery_once7`:
  2 resources/16 bits/16 screens/4 comparator decisions.
- `ada_half_nn_fixed_s3_aligned_f16_d768_in_confirmation_once7`:
  2 resources/8 screens/1 decision.

Attempt1 passed compilation but the wrapper wrongly expected `cuda_suite::`
prefixes. List validation stopped before GPU work. Attempt2 used authoritative
bare names from the same binary, with no source change/rebuild/valid rerun.

Measured main `0ccb32d6bde913ba6d55a511442271c14a1ae77448a93b1f6e29fb7c4353b45f`.
[NT128 raw](evidence/attempt2/once7-cuda132-nt128/test.log), SHA256
`cbec55791d1d7fc0ea58f7c306967811a8e9133b9cac9f6594646e256bd66705`.
[F16 NN raw](evidence/attempt2/once7-cuda132-f16-d768-in/test.log), SHA256
`233cc8100ed8a8e83782e53c946aa595a9727ded5ab90d557c9feb996b79f01b`.
No production integration or full/toolkit gates.
