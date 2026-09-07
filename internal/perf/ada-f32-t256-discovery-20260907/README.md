# Ada exact-F32 T256 discovery

Status: STOP at the first live resource gate. No timing, 21-window extension,
production integration, selector change, or second candidate was run.

The only new source surface is
`tests/gemm_bi_fixed_f32_t256_discovery.rs`. It pins the complete existing
`kernels/gemm_bi_fixed/sm120_f32_n64_copyplan.cu` SHA-256
`5015fbfcd457e92759f2f37a9093175d7e37c941fdd1d8d91258fedb06dddb84`
and extracts the unique T256 block between fixed sentinels. The extracted
17,154-byte block SHA-256 is
`8db567604e13ff8afba012235b59b58c6579b5c4857be4bb7164dbb639a38541`.
Only its export is renamed to a test-only SM89 discovery symbol. The frozen
CUDA13.2 test binary SHA-256 was
`121231b5f683977b8ef0d9b1ee6121f574c0ff8b959aee90611c5490b2f0c1b3`.

Focused TDD preserved the initial three assertion failures in `red1.log`, two
compiler diagnostics in `green1.log` and `green2.log`, and the final
warning-free three-pass/two-ignored result in `green3.log`.

The actual CUDA13.2/CC8.9 resource result was:

- registers/thread: 80 (within the at-most-85 gate);
- active CTAs/SM at block256/static32KiB: 3;
- static shared bytes: 32,768;
- local bytes: **24**, violating the required zero-local-memory gate.

The ignored correctness test therefore exited101 before bit, prefix, or tail
cases. `correctness1/` preserves its quiet PRE, raw failure, exit receipt,
POST, and the immediate RELEASE snapshot that failed quietness at9%/1% while
the GPU drained. `correctness1-release/release.json` is the separate later
quiet0%/0%, no-app release receipt. The failed release was not overwritten and
the GPU workload was not rerun.

This is an unqualified resource STOP, not a measured timing loss. The dormant
timing entry was never executed: its two-order helper retains bracket averages
rather than all four physical observations, pools orders for its final boolean,
and checks inputs but not post-timing output bits. Main identified those limits
before any21 decision. No timing-evidence repair or repeat build was needed once
the required zero-local gate failed. Do not use that dormant entry as promotion
proof or interpret the three host passes as GPU bit-correctness passes.

`manifest-selected.sha256` covers the frozen test/source and all raw evidence.
