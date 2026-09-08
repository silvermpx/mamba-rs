# Ada half NT direct-epilogue discovery

CUDA13.2 / RTX6000Ada. Valid loss: keep the retained Fixed-S3 B-XOR vector
epilogue. Removing its shared output exchange is 14.5–15.6% slower than
retained S3, with no Fast-winning stratum. Do not repeat this candidate.

| Dtype / comparator | Eager candidate/comparator p50 | Graph p50 | Worst p95 |
| --- | ---: | ---: | ---: |
| F16 / retained S3 | 1.1553–1.1561 | 1.1450–1.1452 | 1.1576 |
| BF16 / retained S3 | 1.1531–1.1542 | 1.1449–1.1459 | 1.1574 |
| F16 / native-half Fast | 1.1543–1.1589 | 1.1783–1.1814 | 1.2006 |
| BF16 / native-half Fast | 1.2080–1.2087 | 1.1942–1.1993 | 1.2256 |

The exact ignored test
`ada_half_nt_fixed_s3_direct_epilogue_d768_out_vs_retained_and_fast_discovery_once7`
passed. Both half types match forced TC64 and retained S3 exactly in focused
eager/graph repetitions. Candidate resources: 167 registers, zero local/static
shared bytes, 98,304 dynamic shared bytes, 256 threads and occupancy one.
The target is d768-out `(2048,1536,768)`; 20 GEMMs per observation with
256B-aligned logical pointers. This is discovery, not public AUTO admission.

Root replayed all 112 brackets / 448 observations using the recorded ABBA/
BAAB order and nearest-rank quantiles: 16 screens, 24 bit records, two resource
records and four STOP decisions. Native adapter tests passed 7/7. Raw GPU
test and stable private-cache receipts are in `evidence/cuda132/`; PRE and
DRAIN are quiet with no competing application. Immediate RELEASE shows
10% GPU utilization after the test, also with no application.

Exact measured sources are retained by the commit containing this report;
the live harness may subsequently advance to the separate half-TN experiment.

- Main SHA-256: `ada3769cd0b1d47d478e01550419c06818f4ec643a97208b2a00fd6ed8b89ca2`
- Direct helper SHA-256: `087678c83e2e170f25eac6712abb913cbfeac563177b8d1ab054dbbb35947323`
- Parent NT helper SHA-256: `2c41646460a23f92f43417b08b4389bc306386b26d521282c904881b2e1c2fc0`
- Binary SHA-256: `b59ef7576e6c4a31ee10fba018726fc1014139aa0398c887683e32463e17b069`
- Raw log SHA-256: `729260fc69d4278bb64184f4a85b989534e00bf4150bc60fdefb24883f599423`

No production route, Fixed inference, SM120 route, other toolkit or full gate
was changed or qualified by this screen.
