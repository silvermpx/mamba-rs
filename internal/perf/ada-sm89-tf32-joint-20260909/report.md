# Ada SM89 TF32 joint qualification

Date: 2026-09-09/10

Board: NVIDIA RTX 6000 Ada Generation, CC 8.9, 142 SM

Driver API: 13.20
Toolkits: CUDA 12.8, 13.0 and 13.2

## Decision

The production dispatcher admits the exact measured winner for each toolkit:

| Logical cell `(M,K,N)` | CUDA 12.8 | CUDA 13.0 | CUDA 13.2 |
| --- | --- | --- | --- |
| TN `(2048,768,3072)` | joint N96 | joint N96 | joint N96 |
| TN `(2048,1536,768)` | joint N96 | joint N96 | joint N96 |
| TN `(4621,384,1928)` | joint M64N64 | joint M64N64 | joint M64N64 |
| NN `(4621,384,1928)` | portable M128N128/S3 | portable M128N128/S3 | joint direct N96 |
| NN `(2048,1536,768)` | joint N96 | joint N96 | joint N96 |

CUDA 12.8 and 13.0 deliberately retain the portable NN Prism route. The joint
candidate was 1.8-2.6% slower there, so promoting all five rows would have been
a regression. CUDA 13.2 admits the joint NN Prism route because it passed both
orders and both execution paths.

## Pre-admission performance

The comparator is the previously selected portable TF32 route. Each accepted
row passed ABBA and BAAB, eager and CUDA Graph, first with three screening
windows and then seven official windows. Admission required candidate/prior
portable p50 and p95 below `0.99`.

| Toolkit | Cell | Official candidate/prior range | Result |
| --- | --- | ---: | --- |
| 12.8 | TN d768-in | 0.655935-0.659745 | admit joint |
| 12.8 | TN d768-out | 0.551306-0.553219 | admit joint |
| 12.8 | TN Prism | 0.708787-0.712877 | admit joint |
| 12.8 | NN d768-out | 0.864689-0.865784 | admit joint |
| 12.8 | NN Prism | 1.0186-1.0259 screen | retain portable |
| 13.0 | TN d768-in | 0.655930-0.659921 | admit joint |
| 13.0 | TN d768-out | 0.551353-0.553445 | admit joint |
| 13.0 | TN Prism | 0.708766-0.713023 | admit joint |
| 13.0 | NN d768-out | 0.864669-0.865766 | admit joint |
| 13.0 | NN Prism | 1.0186-1.0260 screen | retain portable |
| 13.2 | TN d768-in | 0.650379-0.653422 | admit joint |
| 13.2 | TN d768-out | 0.546292-0.549530 | admit joint |
| 13.2 | TN Prism | 0.709868-0.714903 | admit joint |
| 13.2 | NN d768-out | 0.819188-0.826986 | admit joint |
| 13.2 | NN Prism | 0.968954-0.975172 | admit joint |

## Correctness and binding gates

- All five forced routes passed eager/graph manifest equality, exact bit
  equality with the retained deterministic route, repeated-bit checks,
  input immutability and red-zone validation on all three toolkits.
- The joint module passed exact export, Driver ABI, PTX/SASS and resource
  checks on all three toolkits. CUDA 12.8/13.0 use 131/131/135 registers for
  the three N96 bodies; CUDA 13.2 uses 124/124/127. All have zero stack, local
  memory and spills. The TN M64N64 and transpose use 83 and 26 registers.
- AUTO cohort matching freezes module, compiler, artifact, header, NVRTC
  library, driver, device and capability identity. Lower-toolkit NN Prism also
  requires its exact portable-module twin; a missing or changed twin fails
  closed.
- Native dispatcher checks cover all 29 single-field identity mutations,
  neighboring dimensions and strides, invalid pointers, alignment, epilogue
  and bias drift.

## Frozen joint identities

| Toolkit | compile/invocation | artifact | NVRTC library domain |
| --- | --- | --- | --- |
| 12.8 | `7fc5e84f898765416780218e633ad40693a57df8b10c8e26a49ef7e5a92cc1e6` | `fe073fa382837907e8813ed82d1af794ee7c615237c27095d4234ea85ea6b701` | `26b0a3a02044ffcbc1693fd83e9261beffa692a4fbcfe3ac5e9d8c87980bb155` |
| 13.0 | `597e1c032bb03fe940f5c472da03d7d7cf83e70f208c064b4efb3de4a358bedc` | `3551770279703636d8c09704bab83bd077bb9894eb4552b3afc3d7e55a3f2e13` | `709b91c36bfb0ed966ee69adc8d6f87ff110eecf3dfb5060367f183ce614eb0d` |
| 13.2 | `50671423cfddb180ee60c618e638d0b1d396ecfb1203364f338c0e934b4a1381` | `f9b43258813e70331206c1e042d1daa62f5f60143761f792a61c016dde1736ef` | `d031a53eb97235b70f62f652932db1bdf728ea229c8ca809d53c5ffd91642687` |

All three share source digest
`ae4b432ccde278a0c4d4a7a742fc4a7c9e503298b01f98360356a42334f90355`,
header manifest digest
`0b64102d321829920d022e321ad0d13a722f33299378c0e9d65476c886f90e59`,
and driver-build digest
`d1edc5a5bc3e10a2688e21568dccbde5e280dc39144dd81e853843ca98b2d0e1`.

## Raw evidence

- `tf32-joint-pre-admission-cuda128.log`
- `tf32-joint-pre-admission-cuda130.log`
- `tf32-joint-pre-admission-cuda132.log`
- `tf32-joint-post-admission-cuda128.log`
- `tf32-joint-post-admission-cuda130.log`
- `tf32-joint-post-admission-cuda132.log`

The post-admission logs are the final proof that public deterministic AUTO
selects this toolkit-specific map in both eager and CUDA Graph execution.
