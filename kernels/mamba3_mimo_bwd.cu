// Mamba-3 MIMO training on the GPU, the sequential path: the backward of
// the scan in mamba3_mimo.cu, the B and C row norms, the angle projection
// and the MLP gate.
//
// The scan backward runs one block per (sequence, head), like the scan.
// It walks the window's segments from the last: each segment is replayed
// from the checkpoint the forward wrote, through the same token pieces
// (so the replay is the forward's bits), recording each token's state
// row, keys, queries, rank inputs and readouts; then the segment's tokens
// are walked back. The state gradient dS (rows in shared memory), the
// previous keys' gradient, the previous input's gradient and the angles'
// gradient carry from token to token and from segment to segment, starting
// from zero after the window (the gradients stop at the states a window
// starts from).
//
// Per token, walked back (p a channel, r a rank, n a state index):
//   gate      dy_r = dO Phi_r silu(Z_r z),   dz, dPhi, dZ
//   readout   du_r = D dy_r,  dS[p] += sum_r dy_r q_r,  dq_r = sum_p dy_r s[p]
//   state     dalpha = sum dS s_prev, dS <- alpha dS (the carry),
//             dkc_r += sum_p dS gamma u_r, dkp_r = sum_p dS beta up_r,
//             dbeta = sum dS kp up, dgamma = sum dS kc u, du_r, dup_r
//   inputs    dx = carry + sum_r du_r Psi_r, carry <- sum_r dup_r Psi_r, dPsi
//   rotation  back to the biased keys and queries, dtheta per angle
//   angles    dtheta carries; d tanh_pi = dtheta dt; ddt += dtheta tanh_pi
//   scalars   alpha = e^(A dt), beta = alpha dt lam_c, gamma = lam dt
//
// Every reduction is a fixed order or a fixed tree; per-sequence weight
// partials are summed over the sequences ascending by m3m_sum_rows.
//
// Composed after mamba3_mimo.cu in the MIMO module.

__device__ __forceinline__ float mimo_silu_grad(float v) {
    float sg = mimo_sigmoid(v);
    return __fmul_rn(sg, __fmaf_rn(v, __fsub_rn(1.0f, sg), 1.0f));
}

// A block-wide sum of the values of the first `n` threads: their values
// into `red`, then a fixed tree over the next power of two. Every thread
// of the block calls it.
__device__ __forceinline__ float mimo_block_sum_n(float v, float* red, int n) {
    const int t = threadIdx.x;
    if (t < n) red[t] = v;
    __syncthreads();
    int width = 1;
    while (width < n) width <<= 1;
    for (int w = width >> 1; w > 0; w >>= 1) {
        if (t < w && t + w < n) red[t] = __fadd_rn(red[t], red[t + w]);
        __syncthreads();
    }
    float out = red[0];
    __syncthreads();
    return out;
}

// The sum of one value per thread of the block.
__device__ __forceinline__ float mimo_block_sum(float v, float* red) {
    return mimo_block_sum_n(v, red, blockDim.x);
}

// Three sums of the values of the first `n` threads (at least a warp) at
// once, each through mimo_block_sum_n's fixed tree in its own third of
// `red` (3 n floats), the result in thread 0 only: the levels above a warp
// in shared memory, the last five as shuffles in warp 0, which pair the
// same operands in the same order. `red` is free again after the next
// barrier.
__device__ __forceinline__ float3 mimo_block_sum3(float a, float b, float c, float* red, int n) {
    const int t = threadIdx.x;
    if (t < n) {
        red[t] = a;
        red[n + t] = b;
        red[2 * n + t] = c;
    }
    __syncthreads();
    int width = 1;
    while (width < n) width <<= 1;
    for (int w = width >> 1; w >= 32; w >>= 1) {
        if (t < w && t + w < n) {
            red[t] = __fadd_rn(red[t], red[t + w]);
            red[n + t] = __fadd_rn(red[n + t], red[n + t + w]);
            red[2 * n + t] = __fadd_rn(red[2 * n + t], red[2 * n + t + w]);
        }
        __syncthreads();
    }
    float3 out = make_float3(0.0f, 0.0f, 0.0f);
    if (t < 32) {
        float x = red[t], y = red[n + t], z = red[2 * n + t];
#pragma unroll
        for (int w = 16; w > 0; w >>= 1) {
            const float xo = __shfl_down_sync(0xffffffffu, x, w);
            const float yo = __shfl_down_sync(0xffffffffu, y, w);
            const float zo = __shfl_down_sync(0xffffffffu, z, w);
            if (t < w && t + w < n) {
                x = __fadd_rn(x, xo);
                y = __fadd_rn(y, yo);
                z = __fadd_rn(z, zo);
            }
        }
        out = make_float3(x, y, z);
    }
    return out;
}

// A 4-byte asynchronous copy into shared memory and its fences.
__device__ __forceinline__ void mimo_cp4(float* dst, const float* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void mimo_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void mimo_cp_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// The record of one token, in floats: the state row after the token
// [P, N], the rotated keys and queries and the biased ones before the
// rotation [R, N] each, u, up and y [P, R], the previous input [P], the
// sines and cosines [na] each, and the step scalars (8).
__device__ __forceinline__ long long mimo_rec_size(int hd, int ds, int rank, int na) {
    return (long long)hd * ds + 4LL * rank * ds + 3LL * hd * rank + hd + 2LL * na + 8;
}

template <typename T, int RK>
__device__ __forceinline__ void mimo_scan_bwd_body(
    T* __restrict__ d_proj, long long ld_proj,
    float* __restrict__ d_bn_h, float* __restrict__ d_cn_h,
    float* __restrict__ d_tanh_h,
    float* __restrict__ g_x, float* __restrict__ g_z, float* __restrict__ g_o,
    float* __restrict__ g_d, float* __restrict__ g_dtb,
    float* __restrict__ g_bb, float* __restrict__ g_cb,
    float* __restrict__ rec,
    const T* __restrict__ d_o,
    const T* __restrict__ proj,
    const float* __restrict__ bn, const float* __restrict__ cn,
    const float* __restrict__ ck_ssm, const float* __restrict__ ck_k,
    const float* __restrict__ ck_v, const float* __restrict__ ck_ang,
    const float* __restrict__ dt_bias,
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,
    const float* __restrict__ D,
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,
    const float* __restrict__ mimo_o,
    int seq, int stride, int nh, int ng, int hd, int ds, int rank, int na,
    float a_floor
) {
    extern __shared__ float mb_sm[];
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int g = h / (nh / ng);
    const int di = nh * hd;
    const int rgn = rank * ng * ds;
    const int rn = rank * ds;
    const int R = MIMO_R(RK, rank);
    constexpr int RB = MIMO_ROW_BATCH(RK);
    const int heads = 2 * di + 2 * rgn;
    const int ang_col = heads + 3 * nh;
    const long long bh = (long long)b * nh + h;
    const int nseg = (seq + stride - 1) / stride;
    const long long rsz = mimo_rec_size(hd, ds, rank, na);
    float* st = mb_sm;                         // replay state rows [P][N+1]
    const int rows4 = (hd * (ds + 1) + 3) & ~3;  // so the keys start on 16 bytes
    float* dsr = st + rows4;                   // dS rows [P][N+1]
    float* kp = dsr + rows4;                   // replay previous keys [R*N]
    float* kc = kp + rn;
    float* qv = kc + rn;
    float* dkc = qv + rn;                      // this token's key gradient
    float* dq = dkc + rn;
    float* dkcar = dq + rn;                    // the previous keys' gradient carry
    float* gbb = dkcar + rn;
    float* gcb = gbb + rn;
    float* dyS = gcb + rn;                     // [P][R]
    float* bvS = dyS + hd * rank;
    float* gxS = bvS + hd * rank;
    float* dth = gxS + hd * rank;              // this token's angle gradient
    float* dthc = dth + na;                    // the angle gradient carry
    float* tpn = dthc + na;                    // tanh_pi of the token
    float* red = tpn + na;                     // 3 lanes
    // The channel threads, a warp multiple: the width of the block sums.
    // The block may be wider, the rest of its threads joining only the
    // loops over state elements and (r, n) pairs.
    const int lanes = (hd + 31) & ~31;

    const bool own = tid < hd;
    float px[MIMO_RA(RK)], pz[MIMO_RA(RK)], po[MIMO_RA(RK)];
    float gx_loc[MIMO_RA(RK)], gz_loc[MIMO_RA(RK)], go_loc[MIMO_RA(RK)];
    _Pragma("unroll") for (int r = 0; r < R; r++) {
        const long long o = ((long long)h * rank + r) * hd + tid;
        px[r] = own ? mimo_x[o] : 0.0f;
        pz[r] = own ? mimo_z[o] : 0.0f;
        po[r] = own ? mimo_o[o] : 0.0f;
        gx_loc[r] = 0.0f;
        gz_loc[r] = 0.0f;
        go_loc[r] = 0.0f;
    }
    const float dval = D[h];
    float gd_loc = 0.0f, gdtb_loc = 0.0f;
    float dx_carry = 0.0f;
    for (int p = 0; p < hd; p++)
        for (int n = tid; n < ds; n += blockDim.x) dsr[p * (ds + 1) + n] = 0.0f;
    for (int i = tid; i < rn; i += blockDim.x) {
        dkcar[i] = 0.0f;
        gbb[i] = 0.0f;
        gcb[i] = 0.0f;
    }
    for (int a = tid; a < na; a += blockDim.x) dthc[a] = 0.0f;
    __syncthreads();

    for (int seg = nseg - 1; seg >= 0; seg--) {
        const int t0 = seg * stride;
        const int t1 = min(t0 + stride, seq);
        const long long q = ((long long)b * nseg + seg) * nh + h;
        // The replay, from the segment's checkpoint, with mimo_scan's pieces
        // (so the forward's bits), recording each token's scalars, angles,
        // keys and queries before and after the turn, inputs and readouts.
        // With builders past the channels it runs as mimo_scan_pipe_body
        // does: the builders make token t + 1's keys, queries and scalars
        // (and their records) while the channels run token t and record its
        // state row themselves, one barrier a token; the walk's dkc, dq and
        // red are its spare key, query and scalar buffers.
        const int spare = (int)blockDim.x - lanes;
        if (spare >= 32 && spare >= na) {
            mimo_state_load(st, ck_ssm + q * hd * (long long)ds, hd, ds);
            float* kb[3] = {kp, kc, dkc};
            float* qb[2] = {qv, dq};
            for (int i = tid; i < rn; i += blockDim.x) kb[2][i] = ck_k[q * rn + i];
            const bool builder = tid >= lanes;
            const int btid = tid - lanes;
            const bool angle_owner = builder && btid < na;
            float theta = angle_owner ? ck_ang[q * na + btid] : 0.0f;
            float v_prev = own ? ck_v[q * hd + tid] : 0.0f;
            const int half = ds / 2;
            const float dtb = dt_bias[h];
            auto build = [&](int t) {
                const int k = t - t0;
                const long long row = (long long)b * seq + t;
                const T* pr = proj + row * ld_proj;
                float* rr = rec + (bh * stride + k) * rsz;
                float* r_kc = rr + (long long)hd * ds;
                float* r_q = r_kc + rn;
                float* r_kpre = r_q + rn;
                float* r_qpre = r_kpre + rn;
                float* r_sn = r_qpre + rn + 3 * hd * rank + hd;
                float* r_cs = r_sn + na;
                float* r_sc = r_cs + na;
                float* kd = kb[k % 3];
                float* qd = qb[k % 2];
                const MimoStep sc = mimo_step_scalars(pr, heads, nh, h, dtb, a_floor);
                if (btid == 0) {
                    float* sd = red + (k % 2) * 4;
                    sd[0] = sc.alpha;
                    sd[1] = sc.beta;
                    sd[2] = sc.gamma;
                    r_sc[0] = sc.alpha;
                    r_sc[1] = sc.beta;
                    r_sc[2] = sc.gamma;
                    r_sc[3] = sc.dt;
                    r_sc[4] = sc.a;
                    r_sc[5] = sc.lam;
                    r_sc[6] = sc.lam_c;
                }
                const float* bnr = bn + row * rgn;
                const float* cnr = cn + row * rgn;
                if (angle_owner) {
                    const int a = btid;
                    theta = mimo_wrap_add(theta, __fmul_rn(mimo_tanh_pi(pr, ang_col, a), sc.dt));
                    float sa, ca;
                    sincosf(theta, &sa, &ca);
                    r_sn[a] = sa;
                    r_cs[a] = ca;
                    for (int r = 0; r < rank; r++) {
                        const long long src = ((long long)r * ng + g) * ds;
                        const long long bias = ((long long)h * rank + r) * ds;
                        const int i0 = r * ds + a, i1 = r * ds + a + half;
                        const float k0 = __fadd_rn(bnr[src + a], b_bias[bias + a]);
                        const float k1 = __fadd_rn(bnr[src + a + half], b_bias[bias + a + half]);
                        const float q0 = __fadd_rn(cnr[src + a], c_bias[bias + a]);
                        const float q1 = __fadd_rn(cnr[src + a + half], c_bias[bias + a + half]);
                        r_kpre[i0] = k0;
                        r_kpre[i1] = k1;
                        r_qpre[i0] = q0;
                        r_qpre[i1] = q1;
                        const float2 kr = mimo_rot_pair(k0, k1, sa, ca);
                        const float2 qr = mimo_rot_pair(q0, q1, sa, ca);
                        kd[i0] = kr.x;
                        kd[i1] = kr.y;
                        qd[i0] = qr.x;
                        qd[i1] = qr.y;
                        r_kc[i0] = kr.x;
                        r_kc[i1] = kr.y;
                        r_q[i0] = qr.x;
                        r_q[i1] = qr.y;
                    }
                }
                for (int i = btid; i < rn; i += spare) {
                    const int r = i / ds, n = i - r * ds;
                    if (n < na || (n >= half && n - half < na)) continue;
                    const long long src = ((long long)r * ng + g) * ds + n;
                    const long long bias = ((long long)h * rank + r) * ds + n;
                    const float kv = __fadd_rn(bnr[src], b_bias[bias]);
                    const float qn = __fadd_rn(cnr[src], c_bias[bias]);
                    kd[i] = kv;
                    qd[i] = qn;
                    r_kpre[i] = kv;
                    r_qpre[i] = qn;
                    r_kc[i] = kv;
                    r_q[i] = qn;
                }
            };
            if (builder && t1 > t0) build(t0);
            __syncthreads();
            for (int t = t0; t < t1; t++) {
                const int k = t - t0;
                if (own) {
                    const long long row = (long long)b * seq + t;
                    const T* pr = proj + row * ld_proj;
                    float* rr = rec + (bh * stride + k) * rsz;
                    float* r_u = rr + (long long)hd * ds + 4 * rn;
                    float* r_up = r_u + hd * rank;
                    float* r_y = r_up + hd * rank;
                    float* r_xp = r_y + hd * rank;
                    const float* sd = red + (k % 2) * 4;
                    const float x = to_f(pr[di + h * hd + tid]);
                    float u[MIMO_RA(RK)], up[MIMO_RA(RK)], y[MIMO_RA(RK)];
                    float* srow = st + tid * (ds + 1);
                    mimo_tok_row<RK>(srow, y, u, up, px, x, v_prev, kb[k % 3], kb[(k + 2) % 3],
                                     qb[k % 2], sd[0], sd[1], sd[2], dval, rank, ds);
                    _Pragma("unroll") for (int r = 0; r < R; r++) {
                        r_u[tid * rank + r] = u[r];
                        r_up[tid * rank + r] = up[r];
                        r_y[tid * rank + r] = y[r];
                    }
                    r_xp[tid] = v_prev;
                    v_prev = x;
                    // The channel's state row after the token into the record,
                    // a float4 at a time where the record row is aligned.
                    float* rs = rr + (long long)tid * ds;
                    int n = 0;
                    if (((ds & 3) | (reinterpret_cast<unsigned long long>(rs) & 15)) == 0) {
                        for (; n + 4 <= ds; n += 4)
                            *reinterpret_cast<float4*>(rs + n) =
                                make_float4(srow[n], srow[n + 1], srow[n + 2], srow[n + 3]);
                    }
                    for (; n < ds; n++) rs[n] = srow[n];
                } else if (builder && t + 1 < t1) {
                    build(t + 1);
                }
                __syncthreads();
            }
        } else {
            mimo_state_load(st, ck_ssm + q * hd * (long long)ds, hd, ds);
            for (int i = tid; i < rn; i += blockDim.x) kp[i] = ck_k[q * rn + i];
            float theta = tid < na ? ck_ang[q * na + tid] : 0.0f;
            float v_prev = own ? ck_v[q * hd + tid] : 0.0f;
            float* kprev = kp;
            float* kcur = kc;
            const int half = ds / 2;
            const float dtb = dt_bias[h];
            __syncthreads();
            for (int t = t0; t < t1; t++) {
                const long long row = (long long)b * seq + t;
                const T* pr = proj + row * ld_proj;
                float* rr = rec + (bh * stride + (t - t0)) * rsz;
                float* r_s = rr;
                float* r_kc = r_s + (long long)hd * ds;
                float* r_q = r_kc + rn;
                float* r_kpre = r_q + rn;
                float* r_qpre = r_kpre + rn;
                float* r_u = r_qpre + rn;
                float* r_up = r_u + hd * rank;
                float* r_y = r_up + hd * rank;
                float* r_xp = r_y + hd * rank;
                float* r_sn = r_xp + hd;
                float* r_cs = r_sn + na;
                float* r_sc = r_cs + na;
                const MimoStep sc = mimo_step_scalars(pr, heads, nh, h, dtb, a_floor);
                const float dt = sc.dt;
                if (tid == 0) {
                    r_sc[0] = sc.alpha;
                    r_sc[1] = sc.beta;
                    r_sc[2] = sc.gamma;
                    r_sc[3] = sc.dt;
                    r_sc[4] = sc.a;
                    r_sc[5] = sc.lam;
                    r_sc[6] = sc.lam_c;
                }
                const float* bnr = bn + row * rgn;
                const float* cnr = cn + row * rgn;
                if (tid < na) {
                    const int a = tid;
                    theta = mimo_wrap_add(theta, __fmul_rn(mimo_tanh_pi(pr, ang_col, a), dt));
                    float sa, ca;
                    sincosf(theta, &sa, &ca);
                    r_sn[a] = sa;
                    r_cs[a] = ca;
                    for (int r = 0; r < rank; r++) {
                        const long long src = ((long long)r * ng + g) * ds;
                        const long long bias = ((long long)h * rank + r) * ds;
                        const int i0 = r * ds + a, i1 = r * ds + a + half;
                        const float k0 = __fadd_rn(bnr[src + a], b_bias[bias + a]);
                        const float k1 = __fadd_rn(bnr[src + a + half], b_bias[bias + a + half]);
                        const float q0 = __fadd_rn(cnr[src + a], c_bias[bias + a]);
                        const float q1 = __fadd_rn(cnr[src + a + half], c_bias[bias + a + half]);
                        r_kpre[i0] = k0;
                        r_kpre[i1] = k1;
                        r_qpre[i0] = q0;
                        r_qpre[i1] = q1;
                        const float2 kr = mimo_rot_pair(k0, k1, sa, ca);
                        const float2 qr = mimo_rot_pair(q0, q1, sa, ca);
                        kcur[i0] = kr.x;
                        kcur[i1] = kr.y;
                        qv[i0] = qr.x;
                        qv[i1] = qr.y;
                        r_kc[i0] = kr.x;
                        r_kc[i1] = kr.y;
                        r_q[i0] = qr.x;
                        r_q[i1] = qr.y;
                    }
                }
                for (int i = tid; i < rn; i += blockDim.x) {
                    const int r = i / ds, n = i - r * ds;
                    if (n < na || (n >= half && n - half < na)) continue;
                    const long long src = ((long long)r * ng + g) * ds + n;
                    const long long bias = ((long long)h * rank + r) * ds + n;
                    const float kv = __fadd_rn(bnr[src], b_bias[bias]);
                    const float qn = __fadd_rn(cnr[src], c_bias[bias]);
                    kcur[i] = kv;
                    qv[i] = qn;
                    r_kpre[i] = kv;
                    r_qpre[i] = qn;
                    r_kc[i] = kv;
                    r_q[i] = qn;
                }
                __syncthreads();
                if (own) {
                    const float x = to_f(pr[di + h * hd + tid]);
                    float u[MIMO_RA(RK)], up[MIMO_RA(RK)], y[MIMO_RA(RK)];
                    float* srow = st + tid * (ds + 1);
                    mimo_tok_row<RK>(srow, y, u, up, px, x, v_prev, kcur, kprev, qv,
                                 sc.alpha, sc.beta, sc.gamma, dval, rank, ds);
                    _Pragma("unroll") for (int r = 0; r < R; r++) {
                        r_u[tid * rank + r] = u[r];
                        r_up[tid * rank + r] = up[r];
                        r_y[tid * rank + r] = y[r];
                    }
                    r_xp[tid] = v_prev;
                    v_prev = x;
                }
                __syncthreads();
                mimo_state_store(r_s, st, hd, ds);
                float* used = kprev;
                kprev = kcur;
                kcur = used;
            }

        }

        // The segment walked back.
        for (int t = t1 - 1; t >= t0; t--) {
            const long long row = (long long)b * seq + t;
            const T* pr = proj + row * ld_proj;
            T* dp = d_proj + row * ld_proj;
            const int slot = t - t0;
            const float* rr = rec + (bh * stride + slot) * rsz;
            const float* r_s = rr;
            const float* r_kc = r_s + (long long)hd * ds;
            const float* r_q = r_kc + rn;
            const float* r_kpre = r_q + rn;
            const float* r_qpre = r_kpre + rn;
            const float* r_u = r_qpre + rn;
            const float* r_up = r_u + hd * rank;
            const float* r_y = r_up + hd * rank;
            const float* r_xp = r_y + hd * rank;
            const float* r_sn = r_xp + hd;
            const float* r_cs = r_sn + na;
            const float* r_sc = r_cs + na;
            const float* prev = slot > 0 ? rec + (bh * stride + slot - 1) * rsz : nullptr;
            const float* s_prev = prev ? prev : ck_ssm + q * hd * (long long)ds;
            const float* k_prev = prev ? prev + (long long)hd * ds : ck_k + q * rn;
            const float alpha = r_sc[0], beta = r_sc[1], gamma = r_sc[2], dt = r_sc[3];
            const float a_val = r_sc[4], lam = r_sc[5], lam_c = r_sc[6];
            // The token's keys and queries and the previous keys into the
            // replay's key buffers, free during the walk, while the gate runs.
            for (int i = tid; i < rn; i += blockDim.x) {
                mimo_cp4(kc + i, r_kc + i);
                mimo_cp4(kp + i, k_prev + i);
                mimo_cp4(qv + i, r_q + i);
            }
            mimo_cp_commit();

            // Gate, D and the readout: dy, dz, du; dS rows take sum_r dy q_r.
            float du[MIMO_RA(RK)], dup[MIMO_RA(RK)];
            if (own) {
                const float z = to_f(pr[h * hd + tid]);
                const float d_o_v = to_f(d_o[row * di + h * hd + tid]);
                float dz = 0.0f;
                _Pragma("unroll") for (int r = 0; r < R; r++) {
                    const float v = __fmul_rn(pz[r], z);
                    const float sg = mimo_sigmoid(v);
                    const float silu = __fmul_rn(v, sg);
                    const float yv = r_y[tid * rank + r];
                    go_loc[r] = __fmaf_rn(d_o_v, __fmul_rn(yv, silu), go_loc[r]);
                    const float dyv = __fmul_rn(__fmul_rn(d_o_v, po[r]), silu);
                    dyS[tid * rank + r] = dyv;
                    const float dv = __fmul_rn(__fmul_rn(__fmul_rn(d_o_v, po[r]), yv),
                                               mimo_silu_grad(v));
                    gz_loc[r] = __fmaf_rn(dv, z, gz_loc[r]);
                    dz = __fmaf_rn(dv, pz[r], dz);
                    du[r] = __fmul_rn(dyv, dval);
                    dup[r] = 0.0f;
                    gd_loc = __fmaf_rn(dyv, r_u[tid * rank + r], gd_loc);
                    bvS[tid * rank + r] = __fmul_rn(beta, r_up[tid * rank + r]);
                    gxS[tid * rank + r] = __fmul_rn(gamma, r_u[tid * rank + r]);
                }
                mimo_put(&dp[h * hd + tid], dz);
            }
            mimo_cp_wait();
            __syncthreads();
            // dS[p][n] += sum_r dy[p][r] q_r[n], each element its own chain,
            // the elements spread over the whole block.
            {
                int p = tid / ds, n = tid - (tid / ds) * ds;
                for (int e = tid; e < hd * ds; e += blockDim.x) {
                    float* d = dsr + p * (ds + 1) + n;
                    float acc = *d;
                    _Pragma("unroll") for (int r = 0; r < R; r++)
                        acc = __fmaf_rn(dyS[p * rank + r], qv[r * ds + n], acc);
                    *d = acc;
                    n += blockDim.x;
                    while (n >= ds) {
                        n -= ds;
                        p++;
                    }
                }
            }
            __syncthreads();
            // Per (r, n): dq from the state after the token, which the replay
            // rows already hold: the replay leaves the segment's last state
            // there, and every earlier token's is the state before the token
            // walked just before it.
            // Two pairs a thread, their chains interleaved.
            for (int i0 = tid; i0 < rn; i0 += 2 * blockDim.x) {
                const int i1 = i0 + blockDim.x;
                const bool two = i1 < rn;
                const int r0 = i0 / ds, n0 = i0 % ds;
                const int r1 = two ? i1 / ds : r0, n1 = two ? i1 % ds : n0;
                float aq0 = 0.0f, aq1 = 0.0f;
                for (int p = 0; p < hd; p++) {
                    const float* srow = st + p * (ds + 1);
                    aq0 = __fmaf_rn(dyS[p * rank + r0], srow[n0], aq0);
                    aq1 = __fmaf_rn(dyS[p * rank + r1], srow[n1], aq1);
                }
                dq[i0] = aq0;
                if (two) dq[i1] = aq1;
            }
            __syncthreads();
            // The state before the token into the replay rows, while the key
            // gradients and the rotation run.
            {
                int p = tid / ds, n = tid - (tid / ds) * ds;
                for (int e = tid; e < hd * ds; e += blockDim.x) {
                    mimo_cp4(st + p * (ds + 1) + n, s_prev + e);
                    n += blockDim.x;
                    while (n >= ds) {
                        n -= ds;
                        p++;
                    }
                }
            }
            mimo_cp_commit();
            // Per (r, n): the previous keys' gradient and the key gradient,
            // two pairs a thread with their chains interleaved.
            for (int i0 = tid; i0 < rn; i0 += 2 * blockDim.x) {
                const int i1 = i0 + blockDim.x;
                const bool two = i1 < rn;
                const int r0 = i0 / ds, n0 = i0 % ds;
                const int r1 = two ? i1 / ds : r0, n1 = two ? i1 % ds : n0;
                float akp0 = 0.0f, akc0 = dkcar[i0];
                float akp1 = 0.0f, akc1 = two ? dkcar[i1] : 0.0f;
                for (int p = 0; p < hd; p++) {
                    const float* drow_ = dsr + p * (ds + 1);
                    const float dsv0 = drow_[n0], dsv1 = drow_[n1];
                    akp0 = __fmaf_rn(dsv0, bvS[p * rank + r0], akp0);
                    akc0 = __fmaf_rn(dsv0, gxS[p * rank + r0], akc0);
                    akp1 = __fmaf_rn(dsv1, bvS[p * rank + r1], akp1);
                    akc1 = __fmaf_rn(dsv1, gxS[p * rank + r1], akc1);
                }
                dkc[i0] = akc0;
                dkcar[i0] = akp0;
                if (two) {
                    dkc[i1] = akc1;
                    dkcar[i1] = akp1;
                }
            }
            __syncthreads();
            // The rotation back to the biased keys and queries, and dtheta.
            const int half = ds / 2;
            for (int a = tid; a < na; a += blockDim.x) {
                const float s = r_sn[a], c = r_cs[a];
                float acc = dthc[a];
                for (int w = 0; w < 2; w++) {
                    float* gv = w == 0 ? dkc : dq;
                    const float* pv = w == 0 ? r_kpre : r_qpre;
                    _Pragma("unroll") for (int r = 0; r < R; r++) {
                        const float g0 = gv[r * ds + a], g1 = gv[r * ds + a + half];
                        const float v0 = pv[r * ds + a], v1 = pv[r * ds + a + half];
                        const float t0v = __fmul_rn(g0, __fsub_rn(__fmul_rn(-s, v0), __fmul_rn(c, v1)));
                        const float t1v = __fmul_rn(g1, __fsub_rn(__fmul_rn(c, v0), __fmul_rn(s, v1)));
                        acc = __fadd_rn(acc, __fadd_rn(t0v, t1v));
                        gv[r * ds + a] = __fmaf_rn(c, g0, __fmul_rn(s, g1));
                        gv[r * ds + a + half] = __fmaf_rn(-s, g0, __fmul_rn(c, g1));
                    }
                }
                dth[a] = acc;
                dthc[a] = acc;
                tpn[a] = mimo_tanh_pi(pr, ang_col, a);
                d_tanh_h[(row * nh + h) * na + a] = __fmul_rn(acc, dt);
            }
            mimo_cp_wait();
            __syncthreads();
            // The recurrence's scalars and the rank inputs; dS takes alpha.
            float da_loc = 0.0f, db_loc = 0.0f, dg_loc = 0.0f;
            // The chains over n ascending, the state indices in batches and
            // the channel's rank inputs held in registers.
            if (own) {
                float* __restrict__ drow = dsr + tid * (ds + 1);
                const float* __restrict__ prow = st + tid * (ds + 1);
                const float* __restrict__ kpr = kp;
                const float* __restrict__ kcr = kc;
                float upv[MIMO_RA(RK)], uv[MIMO_RA(RK)];
                _Pragma("unroll") for (int r = 0; r < R; r++) {
                    upv[r] = r_up[tid * rank + r];
                    uv[r] = r_u[tid * rank + r];
                }
                const bool vec = RB == 4 && mimo_rows_aligned(kpr, kcr, kcr, ds);
                int n = 0;
                for (; RB > 1 && n + RB <= ds; n += RB) {
                    float dsb[RB], pb[RB], kpb[RB][MIMO_RA(RK)], kcb[RB][MIMO_RA(RK)];
                    _Pragma("unroll") for (int j = 0; j < RB; j++) {
                        dsb[j] = drow[n + j];
                        pb[j] = prow[n + j];
                    }
                    if (vec) {
                        _Pragma("unroll") for (int r = 0; r < R; r++) {
                            mimo_split4(kpb, r, kpr + r * ds + n);
                            mimo_split4(kcb, r, kcr + r * ds + n);
                        }
                    } else {
                        _Pragma("unroll") for (int j = 0; j < RB; j++) {
                            _Pragma("unroll") for (int r = 0; r < R; r++) {
                                kpb[j][r] = kpr[r * ds + n + j];
                                kcb[j][r] = kcr[r * ds + n + j];
                            }
                        }
                    }
                    _Pragma("unroll") for (int j = 0; j < RB; j++) {
                        const float dsv = dsb[j];
                        da_loc = __fmaf_rn(dsv, pb[j], da_loc);
                        _Pragma("unroll") for (int r = 0; r < R; r++) {
                            const float dbv = __fmul_rn(dsv, kpb[j][r]);
                            const float dgx = __fmul_rn(dsv, kcb[j][r]);
                            db_loc = __fmaf_rn(dbv, upv[r], db_loc);
                            dup[r] = __fmaf_rn(dbv, beta, dup[r]);
                            dg_loc = __fmaf_rn(dgx, uv[r], dg_loc);
                            du[r] = __fmaf_rn(dgx, gamma, du[r]);
                        }
                        drow[n + j] = __fmul_rn(alpha, dsv);
                    }
                }
                for (; n < ds; n++) {
                    const float dsv = drow[n];
                    da_loc = __fmaf_rn(dsv, prow[n], da_loc);
                    _Pragma("unroll") for (int r = 0; r < R; r++) {
                        const float dbv = __fmul_rn(dsv, kpr[r * ds + n]);
                        const float dgx = __fmul_rn(dsv, kcr[r * ds + n]);
                        db_loc = __fmaf_rn(dbv, upv[r], db_loc);
                        dup[r] = __fmaf_rn(dbv, beta, dup[r]);
                        dg_loc = __fmaf_rn(dgx, uv[r], dg_loc);
                        du[r] = __fmaf_rn(dgx, gamma, du[r]);
                    }
                    drow[n] = __fmul_rn(alpha, dsv);
                }
                const float x = to_f(pr[di + h * hd + tid]);
                const float xp = r_xp[tid];
                float dx = dx_carry, dxp = 0.0f;
                _Pragma("unroll") for (int r = 0; r < R; r++) {
                    gx_loc[r] = __fmaf_rn(du[r], x, __fmaf_rn(dup[r], xp, gx_loc[r]));
                    dx = __fmaf_rn(du[r], px[r], dx);
                    dxp = __fmaf_rn(dup[r], px[r], dxp);
                }
                mimo_put(&dp[di + h * hd + tid], dx);
                dx_carry = dxp;
            }
            for (int i = tid; i < rn; i += blockDim.x) {
                gbb[i] = __fadd_rn(gbb[i], dkc[i]);
                gcb[i] = __fadd_rn(gcb[i], dq[i]);
                d_bn_h[(row * nh + h) * rn + i] = dkc[i];
                d_cn_h[(row * nh + h) * rn + i] = dq[i];
            }
            const float3 sums = mimo_block_sum3(da_loc, db_loc, dg_loc, red, lanes);
            const float d_alpha_s = sums.x, d_beta = sums.y, d_gamma = sums.z;
            if (tid == 0) {
                float d_dt = 0.0f;
                for (int a = 0; a < na; a++) d_dt = __fmaf_rn(dth[a], tpn[a], d_dt);
                const float d_lam_c = __fmul_rn(d_beta, __fmul_rn(alpha, dt));
                const float d_alpha = __fmaf_rn(d_beta, __fmul_rn(dt, lam_c), d_alpha_s);
                d_dt = __fmaf_rn(d_beta, __fmul_rn(alpha, lam_c), d_dt);
                const float d_lam = __fmul_rn(d_gamma, dt);
                d_dt = __fmaf_rn(d_gamma, lam, d_dt);
                const float d_a = __fmul_rn(__fmul_rn(d_alpha, alpha), dt);
                d_dt = __fmaf_rn(__fmul_rn(d_alpha, alpha), a_val, d_dt);
                // lam (1 - lam) is lam * lam_c, with lam_c = sigmoid(-trap)
                // keeping its digits where one minus lam would round to zero.
                const float d_trap = __fmul_rn(__fsub_rn(d_lam, d_lam_c), __fmul_rn(lam, lam_c));
                const float dd_a = to_f(pr[heads + nh + h]);
                const float ht = dd_a >= 0.0f ? __fadd_rn(1.0f, dd_a)
                                              : __fdiv_rn(1.0f, __fsub_rn(1.0f, dd_a));
                const bool clamped = -ht > -a_floor;
                float d_dd_a = 0.0f;
                if (!clamped) {
                    const float slope = dd_a >= 0.0f ? 1.0f
                        : __fdiv_rn(1.0f, __fmul_rn(__fsub_rn(1.0f, dd_a), __fsub_rn(1.0f, dd_a)));
                    d_dd_a = -__fmul_rn(d_a, slope);
                }
                const float spin = __fadd_rn(to_f(pr[heads + h]), dt_bias[h]);
                const float d_sp = spin > 20.0f ? d_dt : __fmul_rn(d_dt, mimo_sigmoid(spin));
                gdtb_loc = __fadd_rn(gdtb_loc, d_sp);
                mimo_put(&dp[heads + h], d_sp);
                mimo_put(&dp[heads + nh + h], d_dd_a);
                mimo_put(&dp[heads + 2 * nh + h], d_trap);
            }
            __syncthreads();
        }
    }

    // The sequence's weight partials.
    if (own) {
        _Pragma("unroll") for (int r = 0; r < R; r++) {
            const long long o = (bh * rank + r) * hd + tid;
            g_x[o] = gx_loc[r];
            g_z[o] = gz_loc[r];
            g_o[o] = go_loc[r];
        }
    }
    for (int i = tid; i < rn; i += blockDim.x) {
        g_bb[bh * rn + i] = gbb[i];
        g_cb[bh * rn + i] = gcb[i];
    }
    const float gd = mimo_block_sum_n(gd_loc, red, lanes);
    if (tid == 0) {
        g_d[bh] = gd;
        g_dtb[bh] = gdtb_loc;
    }
}

#define DEFINE_MIMO_SCAN_BWD(NAME, T)                                           \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj, long long ld_proj,                                  \
    float* __restrict__ d_bn_h, float* __restrict__ d_cn_h,                     \
    float* __restrict__ d_tanh_h,                                               \
    float* __restrict__ g_x, float* __restrict__ g_z, float* __restrict__ g_o,  \
    float* __restrict__ g_d, float* __restrict__ g_dtb,                         \
    float* __restrict__ g_bb, float* __restrict__ g_cb,                         \
    float* __restrict__ rec,                                                    \
    const T* __restrict__ d_o,                                                  \
    const T* __restrict__ proj,                                                 \
    const float* __restrict__ bn, const float* __restrict__ cn,                 \
    const float* __restrict__ ck_ssm, const float* __restrict__ ck_k,           \
    const float* __restrict__ ck_v, const float* __restrict__ ck_ang,           \
    const float* __restrict__ dt_bias,                                          \
    const float* __restrict__ b_bias, const float* __restrict__ c_bias,         \
    const float* __restrict__ D,                                                \
    const float* __restrict__ mimo_x, const float* __restrict__ mimo_z,         \
    const float* __restrict__ mimo_o,                                           \
    int seq, int stride, int nh, int ng, int hd, int ds, int rank, int na,      \
    float a_floor                                                               \
) {                                                                             \
    MIMO_RANK_DISPATCH(rank, mimo_scan_bwd_body<T, RK>(d_proj, ld_proj, d_bn_h, \
        d_cn_h, d_tanh_h, g_x, g_z, g_o, g_d, g_dtb, g_bb, g_cb, rec, d_o,      \
        proj, bn, cn, ck_ssm, ck_k, ck_v, ck_ang, dt_bias, b_bias, c_bias, D,   \
        mimo_x, mimo_z, mimo_o, seq, stride, nh, ng, hd, ds, rank, na,          \
        a_floor))                                                               \
}
DEFINE_MIMO_SCAN_BWD(mimo_scan_bwd, float)
DEFINE_MIMO_SCAN_BWD(mimo_scan_bwd_bf16, __nv_bfloat16)
DEFINE_MIMO_SCAN_BWD(mimo_scan_bwd_f16, __half)

// The B and C row norms backward, one block per (token, rank-group row,
// B or C): the group's heads' gradients summed ascending, then the row
// norm's backward (the forward's sum of squares recomputed in its order):
//   xh = raw inv, c = sum(d w xh) / N, d raw = inv (d w - xh c),
// written into the B or C columns of d_proj, and d xh into dw_parts for
// the norm weights. Grid (rows * R * G, 2), block 128.
template <typename T>
__device__ __forceinline__ void mimo_bc_norm_bwd_body(
    T* __restrict__ d_proj, float* __restrict__ dw_b, float* __restrict__ dw_c,
    const float* __restrict__ d_bn_h, const float* __restrict__ d_cn_h,
    const T* __restrict__ proj, long long ld_proj, int b_col, int c_col,
    const float* __restrict__ wb, const float* __restrict__ wc,
    long long rows, int nh, int ng, int rank, int ds, float eps
) {
    __shared__ float red[128];
    __shared__ float vals[2];
    const int rg = rank * ng;
    const long long q = blockIdx.x;
    if (q >= rows * rg) return;
    const long long row = q / rg;
    const int i = (int)(q % rg);
    const int r = i / ng, g = i % ng;
    const int which = blockIdx.y;
    const int col = which ? c_col : b_col;
    const float* w = which ? wc : wb;
    const float* gh = which ? d_cn_h : d_bn_h;
    float* dw = which ? dw_c : dw_b;
    const T* src = proj + row * ld_proj + col + (long long)i * ds;
    const int hpg = nh / ng;
    float acc = 0.0f;
    for (int n = threadIdx.x; n < ds; n += blockDim.x) {
        float v = to_f(src[n]);
        acc = __fmaf_rn(v, v, acc);
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        vals[0] = __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(red[0], (float)ds), eps)));
    __syncthreads();
    const float inv = vals[0];
    float cacc = 0.0f;
    for (int n = threadIdx.x; n < ds; n += blockDim.x) {
        float d = 0.0f;
        for (int k = 0; k < hpg; k++)
            d = __fadd_rn(d, gh[((row * nh + g * hpg + k) * rank + r) * ds + n]);
        const float xh = __fmul_rn(to_f(src[n]), inv);
        dw[q * ds + n] = __fmul_rn(d, xh);
        cacc = __fmaf_rn(__fmul_rn(d, w[n]), xh, cacc);
    }
    red[threadIdx.x] = cacc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) vals[1] = __fdiv_rn(red[0], (float)ds);
    __syncthreads();
    const float c1 = vals[1];
    for (int n = threadIdx.x; n < ds; n += blockDim.x) {
        float d = 0.0f;
        for (int k = 0; k < hpg; k++)
            d = __fadd_rn(d, gh[((row * nh + g * hpg + k) * rank + r) * ds + n]);
        const float xh = __fmul_rn(to_f(src[n]), inv);
        const float dx = __fmul_rn(inv, __fsub_rn(__fmul_rn(d, w[n]), __fmul_rn(xh, c1)));
        mimo_put(&d_proj[row * ld_proj + col + (long long)i * ds + n], dx);
    }
}

#define DEFINE_MIMO_BC_NORM_BWD(NAME, T)                                        \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj, float* __restrict__ dw_b, float* __restrict__ dw_c, \
    const float* __restrict__ d_bn_h, const float* __restrict__ d_cn_h,         \
    const T* __restrict__ proj, long long ld_proj, int b_col, int c_col,        \
    const float* __restrict__ wb, const float* __restrict__ wc,                 \
    long long rows, int nh, int ng, int rank, int ds, float eps                 \
) {                                                                             \
    mimo_bc_norm_bwd_body<T>(d_proj, dw_b, dw_c, d_bn_h, d_cn_h, proj, ld_proj, \
        b_col, c_col, wb, wc, rows, nh, ng, rank, ds, eps);                     \
}
DEFINE_MIMO_BC_NORM_BWD(mimo_bc_norm_bwd, float)
DEFINE_MIMO_BC_NORM_BWD(mimo_bc_norm_bwd_bf16, __nv_bfloat16)
DEFINE_MIMO_BC_NORM_BWD(mimo_bc_norm_bwd_f16, __half)

// The angle projection backward: d raw = (sum over heads ascending of
// d tanh_pi) pi (1 - tanh(raw)^2), into the angle columns of d_proj.
// Grid 1-D over rows * na.
template <typename T>
__device__ __forceinline__ void mimo_angle_bwd_body(
    T* __restrict__ d_proj, const float* __restrict__ d_tanh_h,
    const T* __restrict__ proj, long long ld_proj, int ang_col,
    long long rows, int nh, int na
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * na) return;
    const long long row = i / na;
    const int a = (int)(i % na);
    float d = 0.0f;
    for (int h = 0; h < nh; h++) d = __fadd_rn(d, d_tanh_h[(row * nh + h) * na + a]);
    const float th = tanhf(to_f(proj[row * ld_proj + ang_col + a]));
    const float slope = __fmul_rn(3.14159265358979f, __fsub_rn(1.0f, __fmul_rn(th, th)));
    mimo_put(&d_proj[row * ld_proj + ang_col + a], __fmul_rn(d, slope));
}

#define DEFINE_MIMO_ANGLE_BWD(NAME, T)                                          \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_proj, const float* __restrict__ d_tanh_h,                 \
    const T* __restrict__ proj, long long ld_proj, int ang_col,                 \
    long long rows, int nh, int na                                              \
) {                                                                             \
    mimo_angle_bwd_body<T>(d_proj, d_tanh_h, proj, ld_proj, ang_col, rows, nh,  \
        na);                                                                    \
}
DEFINE_MIMO_ANGLE_BWD(mimo_angle_bwd, float)
DEFINE_MIMO_ANGLE_BWD(mimo_angle_bwd_bf16, __nv_bfloat16)
DEFINE_MIMO_ANGLE_BWD(mimo_angle_bwd_f16, __half)

// g[i] += sum_b parts[b][i], b ascending: the per-sequence weight partials.
// Grid 1-D over n.
extern "C" __global__ void mimo_sum_rows(
    float* __restrict__ g, const float* __restrict__ parts, int count, long long n
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int b = 0; b < count; b++) acc = __fadd_rn(acc, parts[(long long)b * n + i]);
    g[i] = __fadd_rn(g[i], acc);
}

// The MLP gate backward, act = v silu(gate): dv = d silu(gate),
// d gate = d v silu'(gate), into the up gradient [rows, 2 hidden].
// Grid 1-D over rows * hidden.
template <typename T>
__device__ __forceinline__ void mimo_mlp_gate_bwd_body(
    T* __restrict__ d_up, const T* __restrict__ d_act, const T* __restrict__ up,
    int rows, int hidden
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)rows * hidden;
    if (i >= n) return;
    const long long r = i / hidden;
    const long long j = i - r * hidden;
    const float d = to_f(d_act[i]);
    const float v = to_f(up[r * 2 * hidden + j]);
    const float gte = to_f(up[r * 2 * hidden + hidden + j]);
    const float silu = __fdiv_rn(gte, __fadd_rn(1.0f, mimo_exp(-gte)));
    mimo_put(&d_up[r * 2 * hidden + j], __fmul_rn(d, silu));
    mimo_put(&d_up[r * 2 * hidden + hidden + j], __fmul_rn(__fmul_rn(d, v), mimo_silu_grad(gte)));
}

#define DEFINE_MIMO_MLP_GATE_BWD(NAME, T)                                       \
extern "C" __global__ void NAME(                                                \
    T* __restrict__ d_up, const T* __restrict__ d_act, const T* __restrict__ up,\
    int rows, int hidden                                                        \
) {                                                                             \
    mimo_mlp_gate_bwd_body<T>(d_up, d_act, up, rows, hidden);                   \
}
DEFINE_MIMO_MLP_GATE_BWD(mimo_mlp_gate_bwd, float)
DEFINE_MIMO_MLP_GATE_BWD(mimo_mlp_gate_bwd_bf16, __nv_bfloat16)
DEFINE_MIMO_MLP_GATE_BWD(mimo_mlp_gate_bwd_f16, __half)
