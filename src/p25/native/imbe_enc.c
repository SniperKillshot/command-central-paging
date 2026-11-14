/*
 * P25 Phase 1 IMBE (7200x4400) voice encoder - compiled version.
 *
 * A direct C port of src/p25/imbe.py, which is itself a port of the
 * fixed-point encoder dvmhost ships in src/vocoder/imbe (Pavel Yazev's
 * imbe_vocoder from OP25, built on the ETSI/ITU basic operators). Every step
 * follows the Python operation for operation, with the same 16/32-bit
 * saturation and wrap-around, so both produce identical frames
 * (tests/test_imbe_native.py checks this).
 *
 * The known-tone pitch choice (imbe.py tone_hint) stays in Python; the
 * caller passes the chosen pitch index and the tone harmonics in.
 *
 * Build: python tools/build_native.py
 *
 * GPL-2.0 (derived from dvmhost / OP25).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include "imbe_tables.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#define THREAD_LOCAL __declspec(thread)
#else
#define EXPORT __attribute__((visibility("default")))
#define THREAD_LOCAL __thread
#endif

#define MAX_16 32767
#define MIN_16 (-32768)
#define MAX_32 ((int64_t)2147483647)
#define MIN_32 ((int64_t)-2147483647 - 1)

#define FRAME 160
#define NUM_HARMS_MAX 56
#define NUM_HARMS_MIN 9
#define NUM_BANDS_MAX 12
#define MAX_BLOCK_LEN 10
#define NUM_PRED_RES_BLKS 6
#define PE_LPF_ORD 21
#define PITCH_EST_FRAME 301
#define B_NUM (NUM_HARMS_MAX - 1)
#define FFTLENGTH 256
#define PITCH_EST_BUF_SIZE 621
#define EN_BIT_STREAM_LEN (3 + 3 * 12 + 6 + 2 * 11 + 3)
#define GAIN_QNT_TBL_SIZE 64

#define CNST_0_9254_Q0_16 60647
#define CNST_0_33_Q0_16 0x5556
#define CNST_ONE_Q8_24 0x01000000u
#define CNST_0_7_Q1_15 0x599A
#define CNST_0_4_Q1_15 0x3333
#define CNST_0_03_Q1_15 0x03D7
#define CNST_0_05_Q1_15 0x0666
#define CNST_1_125_Q8_8 0x0120
#define CNST_0_5_Q8_8 0x0080
#define CNST_0_125_Q8_8 0x0020
#define CNST_0_25_Q8_8 0x0040
#define CNST_0_8717_Q1_15 0x6F94
#define CNST_0_0031_Q1_15 0x0064
#define CNST_0_48_Q4_12 0x07AE
#define CNST_1_00_Q4_12 0x1000
#define CNST_0_85_Q4_12 0x0D9B
#define CNST_0_4_Q4_12 0x0666
#define CNST_0_05_Q4_12 0x00CD
#define CNST_0_5882_Q1_15 0x4B4B
#define CNST_0_2857_Q1_15 0x2492
#define CNST_0_5_Q1_15 0x4000
#define CNST_0_5_Q5_11 0x0400
#define CNST_1_0_Q1_15 0x7FFF
#define CNST_0_5_Q0_16 0x8000
#define X05_Q15 16384
#define ONE_Q15 32767

/* ------------------------------------------------------------------------ */
/*  Errors: the C original abort()s on a div_s domain error; here the encode */
/*  call returns -1 and the caller substitutes silence (VocoderError).      */
/* ------------------------------------------------------------------------ */

static THREAD_LOCAL jmp_buf *g_abort;

static void fail(void)
{
    longjmp(*g_abort, 1);
}

/* ------------------------------------------------------------------------ */
/*  Fixed-point basic operators. 16-bit values are held in int, 32-bit in   */
/*  int32_t, with int64_t for intermediates (as Python's unbounded ints).   */
/* ------------------------------------------------------------------------ */

static inline int s16(int64_t x) { return (int16_t)(uint16_t)((uint64_t)x & 0xFFFFu); }
static inline int32_t s32(int64_t x) { return (int32_t)(uint32_t)((uint64_t)x & 0xFFFFFFFFu); }
static inline int sat16(int64_t x) { return x > MAX_16 ? MAX_16 : (x < MIN_16 ? MIN_16 : (int)x); }
static inline int32_t sat32(int64_t x) { return x > MAX_32 ? (int32_t)MAX_32 : (x < MIN_32 ? (int32_t)MIN_32 : (int32_t)x); }

static inline int add(int a, int b) { return sat16((int64_t)a + b); }
static inline int sub(int a, int b) { return sat16((int64_t)a - b); }
static inline int negate(int a) { return a == MIN_16 ? MAX_16 : -a; }

static int shl(int a, int n);

static int shr(int a, int n)
{
    if (n < 0)
        return shl(a, -(n < -16 ? -16 : n));
    if (n >= 15)
        return a < 0 ? -1 : 0;
    return a >> n;
}

static int shl(int a, int n)
{
    int64_t r;
    if (n < 0)
        return shr(a, -(n < -16 ? -16 : n));
    if (n > 15) {
        if (a != 0)
            return a > 0 ? MAX_16 : MIN_16;
        return 0;
    }
    r = (int64_t)a * ((int64_t)1 << n);
    if (r != s16(r))
        return a > 0 ? MAX_16 : MIN_16;
    return (int)r;
}

static inline int mult(int a, int b) { return sat16(((int64_t)a * b) >> 15); }
static inline int mult_r(int a, int b) { return sat16(((int64_t)a * b + 0x4000) >> 15); }

static inline int32_t L_mult(int a, int b)
{
    int64_t p = (int64_t)a * b;
    return p == 0x40000000 ? (int32_t)MAX_32 : (int32_t)(p * 2);
}

static inline int32_t L_add(int32_t a, int32_t b) { return sat32((int64_t)a + b); }
static inline int32_t L_sub(int32_t a, int32_t b) { return sat32((int64_t)a - b); }
static inline int32_t L_mac(int32_t acc, int a, int b) { return sat32((int64_t)acc + L_mult(a, b)); }
static inline int32_t L_msu(int32_t acc, int a, int b) { return sat32((int64_t)acc - L_mult(a, b)); }
static inline int32_t L_deposit_h(int a) { return (int32_t)((int64_t)a * 65536); }
static inline int extract_h(int32_t L) { return s16(L >> 16); }
static inline int extract_l(int32_t L) { return s16(L); }
static inline int L_round(int32_t L) { return extract_h(sat32((int64_t)L + 0x8000)); }

static int32_t L_shl(int32_t L, int n);

static int32_t L_shr(int32_t L, int n)
{
    if (n < 0)
        return L_shl(L, -(n < -32 ? -32 : n));
    if (n >= 31)
        return L < 0 ? -1 : 0;
    return L >> n;
}

static int32_t L_shl(int32_t L, int n)
{
    int64_t v = L;
    if (n <= 0)
        return L_shr(L, -(n < -32 ? -32 : n));
    while (n > 0) {
        if (v > 0x3FFFFFFF)
            return (int32_t)MAX_32;
        if (v < -0x40000000)
            return (int32_t)MIN_32;
        v *= 2;
        n--;
    }
    return (int32_t)v;
}

static int shr_r(int a, int n)
{
    int r;
    if (n > 15)
        return 0;
    r = shr(a, n);
    if (n > 0 && (a & (1 << (n - 1))))
        r += 1;
    return s16(r);
}

static int32_t L_shr_r(int32_t L, int n)
{
    int64_t r;
    if (n > 31)
        return 0;
    r = L_shr(L, n);
    if (n > 0 && ((int64_t)L & ((int64_t)1 << (n - 1))))
        r += 1;
    return s32(r);
}

static int bit_length(uint32_t v)
{
    int n = 0;
    while (v) {
        n++;
        v >>= 1;
    }
    return n;
}

static int norm_s(int a)
{
    if (a == 0)
        return 0;
    if (a == -1)
        return 15;
    if (a < 0)
        a = ~a;
    return 15 - bit_length((uint32_t)a);
}

static int norm_l(int32_t L)
{
    if (L == 0)
        return 0;
    if (L == -1)
        return 31;
    if (L < 0)
        L = ~L;
    return 31 - bit_length((uint32_t)L);
}

static int div_s(int a, int b)
{
    if (a > b || a < 0 || b < 0)
        fail();
    if (b == 0)
        fail();
    if (a == 0)
        return 0;
    if (a == b)
        return MAX_16;
    return (int)(((int64_t)a << 15) / b);
}

static int32_t L_mpy_ls(int32_t L2, int v1)
{
    int sw = shr(extract_l(L2), 1) & 0x7FFF;
    int32_t out = L_shr(L_mult(v1, sw), 15);
    return L_mac(out, v1, extract_h(L2));
}

static int cos_fxp(int x)
{
    int sign = 0, tx, i1, i2, m, ty;
    tx = x < 0 ? negate(x) : x;
    if (tx > X05_Q15) {
        tx = sub(ONE_Q15, tx);
        sign = 1;
    }
    i1 = shr(tx, 7);
    i2 = add(i1, 1);
    if (i1 == 128)
        return 0;
    m = shl(sub(tx, shl(i1, 7)), 8);
    ty = add(COS_TABLE[i1], mult(m, sub(COS_TABLE[i2], COS_TABLE[i1])));
    return sign ? negate(ty) : ty;
}

static int sin_fxp(int x)
{
    int sign = 0, tx, ty;
    if (x < 0) {
        tx = negate(x);
        sign = 1;
    } else {
        tx = x;
    }
    tx = tx > X05_Q15 ? sub(tx, X05_Q15) : sub(X05_Q15, tx);
    ty = cos_fxp(tx);
    return sign ? negate(ty) : ty;
}

/* returns the mantissa, *exp the exponent */
static int32_t sqrt_l_exp(int32_t L, int *exp)
{
    int e, i, a;
    int32_t y;
    if (L <= 0) {
        *exp = 0;
        return 0;
    }
    e = norm_l(L) & 0xFFFE;
    L = L_shl(L, e);
    *exp = e >> 1;
    L = L_shr(L, 9);
    i = extract_h(L);
    L = L_shr(L, 1);
    a = extract_l(L) & 0x7FFF;
    i = sub(i, 16);
    y = L_deposit_h(SQRT_TABLE[i]);
    y = L_msu(y, sub(SQRT_TABLE[i], SQRT_TABLE[i + 1]), a);
    return y;
}

static int32_t Log2(int x)
{
    int exp, i, a;
    int32_t y;
    if (x <= 0)
        return 0;
    exp = norm_s(x);
    x = shl(x, exp);
    i = shr(x, 9);
    a = shl(x & 0x1FF, 6);
    i = sub(i, 32);
    y = L_deposit_h(LOG_TABLE[i]);
    y = L_msu(y, sub(LOG_TABLE[i], LOG_TABLE[i + 1]), a);
    y = L_shr(y, 9);
    exp = sub(12, exp);
    return L_add(y, L_deposit_h(shl(exp, 6)));
}

/* ------------------------------------------------------------------------ */
/*  Quantiser helpers                                                       */
/* ------------------------------------------------------------------------ */

static void get_bit_allocation(int num_harms, int *out)
{
    int offset, k, i;
    if (num_harms == NUM_HARMS_MIN) {
        offset = 0;
    } else {
        int index = num_harms - NUM_HARMS_MIN - 1;
        offset = BIT_ALLOCATION_OFFSET_TBL[index >> 2] + (3 + (index >> 2)) * (index & 0x3);
    }
    k = 0;
    for (i = 0; i < num_harms - 1; i += 4) {
        int tmp = BIT_ALLOCATION_TBL[offset++];
        out[k + 3] = tmp & 0xF;
        out[k + 2] = (tmp >> 4) & 0xF;
        out[k + 1] = (tmp >> 8) & 0xF;
        out[k + 0] = (tmp >> 12) & 0xF;
        k += 4;
    }
}

/* step_size is the unsigned 16-bit value (the Python passes "& 0xFFFF") */
static int32_t deqnt_by_step(int qval, int64_t step_size, int bit_num)
{
    int32_t res;
    if (bit_num == 0)
        return 0;
    res = s32(step_size * (qval - ((int64_t)1 << (bit_num - 1))));
    return L_add(res, (int32_t)((step_size * CNST_0_5_Q0_16) >> 16));
}

static int qnt_by_step(int val, int64_t step_size, int bit_num)
{
    int step16 = s16(step_size);
    int shift = norm_s(step16);
    int tmp = div_s(0x4000, shl(step16, shift));
    int q_index = shr_r(mult(val, tmp), sub(9, shift));
    int max_val = bit_num >= 1 ? s16((int64_t)1 << (bit_num - 1)) : 0;
    int min_val = negate(max_val);
    if (q_index < min_val)
        return 0;
    if (q_index >= max_val)
        return s16(((int64_t)1 << bit_num) - 1);
    return s16((int64_t)max_val + q_index);
}

static int tbl_quant(int val, const int32_t *q_tbl, int size)
{
    int lo = 0, hi = size - 1;
    if (val >= q_tbl[hi])
        return hi;
    if (val <= q_tbl[lo])
        return lo;
    while (hi - lo != 1) {
        int mid = lo + ((hi - lo) >> 1);
        if (q_tbl[mid] > val)
            hi = mid;
        else
            lo = mid;
    }
    return (q_tbl[hi] - val <= val - q_tbl[lo]) ? hi : lo;
}

/* ------------------------------------------------------------------------ */
/*  Encoder state                                                           */
/* ------------------------------------------------------------------------ */

#define FFT_BUTTERFLIES 1024     /* 8 stages x 128 */

typedef struct {
    int prev_pitch, prev_prev_pitch, prev_e_p, prev_prev_e_p;
    int pitch_est_buf[PITCH_EST_BUF_SIZE];
    int pitch_ref_buf[PITCH_EST_BUF_SIZE];
    int pe_lpf_mem[PE_LPF_ORD];
    int32_t dc_rmv_mem;
    int num_harms_prev2;
    int32_t sa_prev2[NUM_HARMS_MAX + 2];
    int v_uv_dsn[NUM_BANDS_MAX];
    int32_t th_max;

    int wr_array[FFTLENGTH / 2 + 1];
    int wi_array[FFTLENGTH / 2 + 1];
    int fft_perm[2 * FFTLENGTH + 1];
    int fft_n_stages;
    int fft_stage_mmax[16];
    int fft_stage_start[17];
    int fft_i[FFT_BUTTERFLIES];
    int fft_wr[FFT_BUTTERFLIES];
    int fft_wi[FFT_BUTTERFLIES];

    /* IMBE_PARAM */
    int e_p_val, pitch, ref_pitch;
    int32_t fund_freq;
    int num_harms, num_bands;
    int p_v_uv_dsn[NUM_HARMS_MAX];
    int b_vec[NUM_HARMS_MAX + 3];
    int bit_alloc[B_NUM + 4];
    int sa[NUM_HARMS_MAX];
    int l_uv;
} imbe_enc;

static void fft_init(imbe_enc *e)
{
    int half = FFTLENGTH >> 1;
    int step = shl(2, norm_s(half));
    int theta = 0, i, j, m, nn, n, mmax, index_step, count, stage;

    for (i = 0; i < half + 1; i++) {
        e->wr_array[i] = cos_fxp(theta);
        e->wi_array[i] = sin_fxp(theta);
        theta = (i >= half - 1) ? ONE_Q15 : add(theta, step);
    }

    /* fft.cpp's bit-reversal permutation */
    nn = FFTLENGTH;
    n = 2 * nn;
    for (i = 0; i <= n; i++)
        e->fft_perm[i] = i;
    j = 1;
    for (i = 1; i < n; i += 2) {
        if (j > i) {
            int t = e->fft_perm[j];
            e->fft_perm[j] = e->fft_perm[i];
            e->fft_perm[i] = t;
            t = e->fft_perm[j + 1];
            e->fft_perm[j + 1] = e->fft_perm[i + 1];
            e->fft_perm[i + 1] = t;
        }
        m = nn;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    /* every butterfly of every stage, with its twiddle */
    mmax = 2;
    index_step = nn;
    count = 0;
    stage = 0;
    while (n > mmax) {
        int istep = mmax << 1, index = 0, wr = ONE_Q15, wi = 0;
        index_step >>= 1;
        e->fft_stage_mmax[stage] = mmax;
        e->fft_stage_start[stage] = count;
        for (m = 1; m < mmax; m += 2) {
            for (i = m; i <= n; i += istep) {
                e->fft_i[count] = i;
                e->fft_wr[count] = wr;
                e->fft_wi[count] = wi;
                count++;
            }
            index += index_step;
            if (index <= FFTLENGTH / 2) {
                wr = e->wr_array[index];
                wi = e->wi_array[index];
            }
        }
        mmax = istep;
        stage++;
    }
    e->fft_n_stages = stage;
    e->fft_stage_start[stage] = count;
}

static void fft(const imbe_enc *e, const int *re_in, int *re, int *im)
{
    int32_t src[2 * FFTLENGTH + 1], data[2 * FFTLENGTH + 1];
    int i, s, k;
    src[0] = 0;
    for (i = 0; i < FFTLENGTH; i++) {
        src[2 * i + 1] = re_in[i];
        src[2 * i + 2] = 0;
    }
    for (i = 0; i <= 2 * FFTLENGTH; i++)
        data[i] = src[e->fft_perm[i]];
    for (s = 0; s < e->fft_n_stages; s++) {
        int mmax = e->fft_stage_mmax[s];
        for (k = e->fft_stage_start[s]; k < e->fft_stage_start[s + 1]; k++) {
            int ii = e->fft_i[k], jj = ii + mmax;
            int wr = e->fft_wr[k], wi = e->fft_wi[k];
            int32_t dr = data[jj], di = data[jj + 1];
            int64_t tr = sat32(((int64_t)L_mult(wr, dr) >> 1) - ((int64_t)L_mult(wi, di) >> 1));
            int64_t ti = sat32(((int64_t)L_mult(wr, di) >> 1) + ((int64_t)L_mult(wi, dr) >> 1));
            int64_t a = (int64_t)data[ii] * 32768;
            int64_t b = (int64_t)data[ii + 1] * 32768;
            data[jj] = L_round(sat32(a - tr));
            data[ii] = L_round(sat32(a + tr));
            data[jj + 1] = L_round(sat32(b - ti));
            data[ii + 1] = L_round(sat32(b + ti));
        }
    }
    for (i = 0; i < FFTLENGTH; i++) {
        re[i] = data[2 * i + 1];
        im[i] = data[2 * i + 2];
    }
}

static void enc_init(imbe_enc *e)
{
    memset(e, 0, sizeof(*e));
    e->prev_pitch = 158;
    e->prev_prev_pitch = 158;
    e->num_harms_prev2 = 30;
    fft_init(e);
}

/* ------------------------------------------------------------------------ */
/*  Front end                                                               */
/* ------------------------------------------------------------------------ */

static void dc_rmv(imbe_enc *e, const int16_t *snd, int *out)
{
    int32_t mem = e->dc_rmv_mem;
    int i;
    for (i = 0; i < FRAME; i++) {
        int32_t t = L_deposit_h(snd[i]);
        mem = L_add(mem, t);
        out[i] = L_round(mem);
        mem = L_mpy_ls(mem, 0x7EB8);
        mem = L_sub(mem, t);
    }
    e->dc_rmv_mem = mem;
}

static void pe_lpf(imbe_enc *e, const int *sig, int *out)
{
    int hist[PE_LPF_ORD - 1 + FRAME];
    int k, t;
    for (k = 0; k < PE_LPF_ORD - 1; k++)
        hist[k] = e->pe_lpf_mem[k + 1];
    for (k = 0; k < FRAME; k++)
        hist[PE_LPF_ORD - 1 + k] = sig[k];
    for (k = 0; k < FRAME; k++) {
        int32_t acc = 0;
        for (t = 0; t < PE_LPF_ORD; t++)
            acc = sat32((int64_t)acc + L_mult(hist[k + t], LPF_COEF[t]));
        out[k] = L_round(acc);
    }
    for (k = 0; k < PE_LPF_ORD; k++)
        e->pe_lpf_mem[k] = hist[FRAME - 1 + k];
}

/* ------------------------------------------------------------------------ */
/*  Pitch estimation                                                        */
/* ------------------------------------------------------------------------ */

static void e_p(const int *sigin, int *res)
{
    int w[PITCH_EST_FRAME];
    int32_t corr[259];
    int32_t L_sum, L_e0;
    int scale, i, r, j;
    int den_part_acc, step, beg;

    for (i = 0; i < PITCH_EST_FRAME; i++)
        w[i] = mult_r(sigin[i], WI[i]);

    L_sum = 0;
    for (i = 0; i < PITCH_EST_FRAME; i++)
        L_sum = L_add(L_sum, L_mpy_ls(L_mult(sigin[i], sigin[i]), WI[i]));
    if (L_sum == MAX_32) {
        L_sum = 0;
        for (i = 0; i < PITCH_EST_FRAME; i++)
            L_sum = L_add(L_sum, L_mpy_ls(L_mult(sigin[i], sigin[i]) >> 5, WI[i]));
        scale = 5;
    } else {
        scale = 0;
    }
    L_e0 = 0;
    for (i = 0; i < PITCH_EST_FRAME; i++)
        L_e0 = L_add(L_e0, L_shr(L_mult(w[i], w[i]), scale));

    for (r = 0; r < 130; r++) {
        int lag = 21 + r;
        int32_t acc = 0;
        for (i = 0; i + lag < PITCH_EST_FRAME; i++)
            acc = L_add(acc, L_shr(L_mult(w[i], w[i + lag]), scale));
        corr[2 * r] = acc;
    }
    for (i = 1; i < 258; i += 2)
        corr[i] = L_shr(L_add(corr[i - 1], corr[i + 1]), 1);

    den_part_acc = CNST_0_8717_Q1_15;
    step = 42;
    beg = 0;
    L_e0 = L_shr(L_e0, 7);
    for (i = 0; i < 203; i++) {
        int32_t L_tmp = 0, L_num, L_den;
        for (j = beg; j <= 258; j += step)
            L_tmp = L_add(L_tmp, corr[j]);
        L_tmp = L_shr(L_tmp, 6);
        L_tmp = L_add(L_tmp, L_e0);
        L_tmp = s32((int64_t)L_tmp * step);
        L_num = L_sub(L_sum, L_tmp);
        beg++;
        step++;
        L_den = L_mpy_ls(L_sum, den_part_acc);
        if (L_num < L_den && L_den != 0) {
            if (L_num <= 0) {
                res[i] = 0;
            } else {
                int t = norm_l(L_den);
                t = div_s(extract_h(L_shl(L_num, t)), extract_h(L_shl(L_den, t)));
                res[i] = shr(t, 3);
            }
        } else {
            res[i] = CNST_1_00_Q4_12;
        }
        den_part_acc = sub(den_part_acc, CNST_0_0031_Q1_15);
    }
}

static void pitch_est(imbe_enc *e, const int *buf)
{
    int arr0[203], arr1[203], arr2[203], arr2_min[203], est_save[203];
    int min_index, max_index, p, pb, e_cur, ceb, p0_est, cef_est, p0, p1, pf, cef;

    e_p(buf, arr0);
    min_index = (MIN_MAX_TBL[e->prev_pitch] >> 8) & 0xFF;
    max_index = MIN_MAX_TBL[e->prev_pitch] & 0xFF;
    p = pb = min_index;
    e_cur = arr0[min_index];
    p++;
    while (p <= max_index) {
        if (arr0[p] < e_cur) {
            e_cur = arr0[p];
            pb = p;
        }
        p++;
    }
    ceb = add(e_cur, add(e->prev_e_p, e->prev_prev_e_p));
    if (ceb <= CNST_0_48_Q4_12) {
        e->prev_prev_pitch = e->prev_pitch;
        e->prev_pitch = pb;
        e->prev_prev_e_p = e->prev_e_p;
        e->prev_e_p = arr0[pb];
        e->pitch = pb + 42;
        e->e_p_val = e->prev_e_p;
        return;
    }

    e_p(buf + FRAME, arr1);
    e_p(buf + 2 * FRAME, arr2);
    p0_est = 0;
    cef_est = s16((int64_t)arr0[0] + arr1[0] + arr2[0]);
    for (p1 = 0; p1 < 203; p1++) {
        int p2 = (MIN_MAX_TBL[p1] >> 8) & 0xFF;
        int p2_max = MIN_MAX_TBL[p1] & 0xFF;
        int s_tmp = arr2[p1];
        while (p2 <= p2_max) {
            if (arr2[p2] < s_tmp)
                s_tmp = arr2[p2];
            p2++;
        }
        arr2_min[p1] = s_tmp;
    }
    for (p0 = 0; p0 < 203; p0++) {
        int est = s16((int64_t)arr1[p0] + arr2_min[p0]);
        int p1_max;
        p1 = (MIN_MAX_TBL[p0] >> 8) & 0xFF;
        p1_max = MIN_MAX_TBL[p0] & 0xFF;
        while (p1 <= p1_max) {
            int v = add(arr1[p1], arr2_min[p1]);
            if (v < est)
                est = v;
            p1++;
        }
        est_save[p0] = est;
        cef = add(arr0[p0], est);
        if (cef < cef_est) {
            cef_est = cef;
            p0_est = p0;
        }
    }

    pf = p0_est;
    if (pf >= 42) {
        int i, p_fp, tmp = 0;
        if (pf < 84)
            i = 1;
        else if (pf < 126)
            i = 2;
        else if (pf < 168)
            i = 3;
        else
            i = 4;
        p_fp = ((pf + 42) << 8) & 0xFFFF;
        while (i) {
            int p_index;
            i--;
            if (i == 0)
                tmp = p_fp >> 1;
            else if (i == 1)
                tmp = (int)((((int64_t)p_fp * 0x5555) & 0xFFFFFFFF) >> 16);
            else if (i == 2)
                tmp = p_fp >> 2;
            else if (i == 3)
                tmp = (int)((((int64_t)p_fp * 0x3333) & 0xFFFFFFFF) >> 16);
            tmp &= 0xFFFF;
            p_index = s16(((tmp + 0x0080) >> 8) - 42);
            cef = add(arr0[p_index], est_save[p_index]);
            if (cef <= CNST_0_85_Q4_12 && mult_r(cef, CNST_0_5882_Q1_15) <= cef_est) {
                pf = p_index;
                break;
            }
            if (cef <= CNST_0_4_Q4_12 && mult_r(cef, CNST_0_2857_Q1_15) <= cef_est) {
                pf = p_index;
                break;
            }
            if (cef <= CNST_0_05_Q4_12) {
                pf = p_index;
                break;
            }
        }
    }

    cef = add(arr0[pf], est_save[pf]);
    p = ceb <= cef ? pb : pf;
    e->prev_prev_pitch = e->prev_pitch;
    e->prev_pitch = p;
    e->prev_prev_e_p = e->prev_e_p;
    e->prev_e_p = arr0[p];
    e->pitch = p + 42;
    e->e_p_val = e->prev_e_p;
}

/* ------------------------------------------------------------------------ */
/*  Pitch refinement                                                        */
/* ------------------------------------------------------------------------ */

static inline int window_index(int32_t L_tmp)
{
    int iw = extract_h(L_tmp);
    if (iw < 0 && (L_tmp & 0xFFFF))
        iw = add(iw, 1);
    return add(iw, 160);
}

static void pitch_ref(imbe_enc *e, const int *re, const int *im)
{
    int sp_re[2 * FFTLENGTH], sp_im[2 * FFTLENGTH];
    int pitch_est = sub(shl(e->pitch, 7), CNST_1_125_Q8_8);
    int32_t L_diff_min = (int32_t)MAX_32;
    int pitch_cand = 0;
    int32_t fund_freq_cand = 0;
    int n;

    memset(sp_re, 0, sizeof(sp_re));
    memset(sp_im, 0, sizeof(sp_im));
    for (n = 0; n < 19; n++) {
        int shift = norm_s(pitch_est);
        int tmp = div_s(0x4000, shl(pitch_est, shift));
        int32_t fund_freq = L_shl(tmp, shift + 11);
        int32_t acc = fund_freq;
        int32_t half = L_shr(fund_freq, 1);
        int32_t acc_a = L_sub(fund_freq, half);
        int32_t acc_b = L_add(fund_freq, half);
        int32_t L_sum;
        int up_lim, index_b = 0, i;

        up_lim = extract_h(L_shr(s32((int64_t)CNST_0_9254_Q0_16 * pitch_est), 1));
        up_lim = sub(up_lim, CNST_0_5_Q8_8);
        up_lim = s16(up_lim & 0xFF00);
        up_lim = shr(mult(up_lim, extract_h(fund_freq)), 1);
        while (index_b <= up_lim) {
            int ha = extract_h(acc_a), hb = extract_h(acc_b);
            int index_a = (ha >> 8) + ((ha & 0xFF) ? 1 : 0);
            index_b = (hb >> 8) + ((hb & 0xFF) ? 1 : 0);
            if (index_b >= 50) {
                int tbl[2 * FFTLENGTH];
                int ntbl = 0, ia, k, hre, him;
                int32_t amp_re = 0, amp_im = 0;
                int32_t L_tmp = L_shl(L_deposit_h(index_a), 8);
                L_tmp = L_sub(L_tmp, acc);
                L_tmp = L_add(L_tmp, 0x00020000);
                L_tmp = L_shr(L_tmp, 2);
                for (ia = index_a; ia < index_b; ia++) {
                    int iw = window_index(L_tmp);
                    tbl[ntbl++] = iw;
                    if (iw >= 0 && iw <= 320) {
                        amp_re = L_mac(amp_re, re[ia], WR_SP[iw]);
                        amp_im = L_mac(amp_im, im[ia], WR_SP[iw]);
                    }
                    L_tmp = L_add(L_tmp, 0x400000);
                }
                hre = extract_h(amp_re);
                him = extract_h(amp_im);
                for (k = 0, ia = index_a; k < ntbl; k++, ia++) {
                    int iw = tbl[k];
                    if (iw < 0 || iw > 320) {
                        sp_re[ia] = sp_im[ia] = 0;
                    } else {
                        sp_im[ia] = mult(mult(him, WR_SP[iw]), 0x6666);
                        sp_re[ia] = mult(mult(hre, WR_SP[iw]), 0x6666);
                    }
                }
            }
            acc_a = L_add(acc_a, fund_freq);
            acc_b = L_add(acc_b, fund_freq);
            acc = L_add(acc, fund_freq);
        }
        L_sum = 0;
        for (i = 50; i <= up_lim && i < FFTLENGTH; i++) {
            int rt = sub(re[i], sp_re[i]);
            int it = sub(im[i], sp_im[i]);
            L_sum = sat32((int64_t)L_sum + L_mult(rt, rt));
            L_sum = sat32((int64_t)L_sum + L_mult(it, it));
        }
        if (L_sum < L_diff_min) {
            L_diff_min = L_sum;
            pitch_cand = pitch_est;
            fund_freq_cand = fund_freq;
        }
        pitch_est = add(pitch_est, CNST_0_125_Q8_8);
    }
    e->ref_pitch = pitch_cand;
    e->fund_freq = fund_freq_cand;
}

/* ------------------------------------------------------------------------ */
/*  Voiced/unvoiced decision                                                */
/* ------------------------------------------------------------------------ */

static int voiced_sa_calc(int32_t num, int den)
{
    int ex;
    int32_t L = sqrt_l_exp(L_mpy_ls(num, den), &ex);
    return extract_h(L_shr(L, ex - 3));
}

static int unvoiced_sa_calc(int32_t num, int den)
{
    int shift = norm_s(den), ex;
    int tmp = div_s(0x4000, shl(den, shift));
    int32_t L = L_shl(L_mpy_ls(num, tmp), shift + 2);
    L = sqrt_l_exp(L, &ex);
    L = L_shr(L, ex - 2 + 6);
    return extract_h(L_mpy_ls(L, 0x4A76));
}

typedef struct {
    int32_t D_num, D_den;
    int dsn_thr, band_cnt, index_b, index_a_save, b1_vec, uv_harms_cnt;
    int32_t M_num[NUM_HARMS_MAX];
    int M_den[NUM_HARMS_MAX];
} vuv_ctx;

static void decide(imbe_enc *e, vuv_ctx *c, int first, int last)
{
    int Dk, k;
    c->b1_vec = s16((int64_t)c->b1_vec << 1);
    if (c->D_den > c->D_num && c->D_den != 0) {
        int t = norm_l(c->D_den);
        Dk = div_s(extract_h(L_shl(c->D_num, t)), extract_h(L_shl(c->D_den, t)));
    } else {
        Dk = MAX_16;
    }
    if (Dk < c->dsn_thr) {
        e->v_uv_dsn[c->band_cnt] = 1;
        c->b1_vec |= 1;
        for (k = first; k < last; k++) {
            e->sa[k] = voiced_sa_calc(c->M_num[k], c->M_den[k]);
            e->p_v_uv_dsn[k] = 1;
        }
    } else {
        e->v_uv_dsn[c->band_cnt] = 0;
        for (k = first; k < last; k++) {
            e->sa[k] = unvoiced_sa_calc(c->M_num[k], s16((int64_t)c->index_b - c->index_a_save));
            e->p_v_uv_dsn[k] = 0;
            c->uv_harms_cnt++;
        }
    }
}

static void v_uv_det(imbe_enc *e, const int *re, const int *im)
{
    vuv_ctx c;
    int32_t fund_freq = e->fund_freq;
    int tmp = shr(add(shr(e->ref_pitch, 1), CNST_0_25_Q8_8), 8);
    int num_harms = extract_h(s32((int64_t)CNST_0_9254_Q0_16 * tmp));
    int num_bands, j, M_fcn, num_harms_cnt;
    int32_t th_lf, th_hf, th0, M_fcn_num, M_fcn_den, fund_fr_step, fund_fr_acc, acc, half, acc_a, acc_b;

    if (num_harms < NUM_HARMS_MIN)
        num_harms = NUM_HARMS_MIN;
    if (num_harms > NUM_HARMS_MAX)
        num_harms = NUM_HARMS_MAX;
    if (num_harms <= 36)
        num_bands = extract_h(s32((int64_t)(num_harms + 2) * CNST_0_33_Q0_16));
    else
        num_bands = NUM_BANDS_MAX;
    e->num_harms = num_harms;
    e->num_bands = num_bands;

    th_lf = 0;
    for (j = 0; j < 64; j++) {
        th_lf = L_mac(th_lf, re[j], re[j]);
        th_lf = L_mac(th_lf, im[j], im[j]);
    }
    th_hf = 0;
    for (j = 64; j < 128; j++) {
        th_hf = L_mac(th_hf, re[j], re[j]);
        th_hf = L_mac(th_hf, im[j], im[j]);
    }
    th0 = L_add(th_lf, th_hf);
    if (th0 > e->th_max)
        e->th_max = L_shr(L_add(e->th_max, th0), 1);
    else
        e->th_max = L_add(L_mpy_ls(e->th_max, 0x7EB8), L_mpy_ls(th0, 0x0148));
    M_fcn_num = L_add(th0, L_mpy_ls(e->th_max, 0x0051));
    M_fcn_den = L_add(th0, L_mpy_ls(e->th_max, 0x0148));
    if (M_fcn_den == 0) {
        M_fcn = 0x2000;
    } else {
        int t = norm_l(M_fcn_den);
        int32_t L_tmp;
        M_fcn_den = L_shl(M_fcn_den, t);
        M_fcn_num = L_shl(M_fcn_num, t);
        M_fcn = div_s(extract_h(M_fcn_num), extract_h(M_fcn_den));
        L_tmp = L_add(L_shl(th_hf, 2), th_hf);
        if (th_lf < L_tmp) {
            t = norm_l(L_tmp);
            M_fcn_den = L_shl(L_tmp, t);
            th_lf = L_shl(th_lf, t);
            t = div_s(extract_h(th_lf), extract_h(M_fcn_den));
            L_tmp = sqrt_l_exp(L_deposit_h(t), &t);
            if (t)
                L_tmp = L_shr(L_tmp, t);
            M_fcn = mult(M_fcn, extract_h(L_tmp));
        }
    }

    fund_fr_step = L_shl(L_mpy_ls(fund_freq, 0x6488), 2);
    memset(&c, 0, sizeof(c));
    num_harms_cnt = 0;
    fund_fr_acc = 0;
    acc = fund_freq;
    half = L_shr(fund_freq, 1);
    acc_a = L_sub(fund_freq, half);
    acc_b = L_add(fund_freq, half);

    for (j = 0; j < num_harms; j++) {
        int ha = extract_h(acc_a), hb = extract_h(acc_b);
        int index_a = (ha >> 8) + ((ha & 0xFF) ? 1 : 0);
        int tbl[2 * FFTLENGTH];
        int ntbl = 0, ia, k, sc_coef, im2, re2;
        int32_t L_tmp, M_den_sum = 0, amp_re = 0, amp_im = 0, M_num_sum = 0;

        c.index_b = (hb >> 8) + ((hb & 0xFF) ? 1 : 0);
        L_tmp = L_shl(L_deposit_h(index_a), 8);
        L_tmp = L_sub(L_tmp, acc);
        L_tmp = L_add(L_tmp, 0x00020000);
        L_tmp = L_shr(L_tmp, 2);
        c.index_a_save = index_a;
        if (num_harms_cnt == 0) {
            if (e->e_p_val > 0x08CD && c.band_cnt >= 1)
                c.dsn_thr = 0;
            else if (e->v_uv_dsn[c.band_cnt] == 1)
                c.dsn_thr = mult(M_fcn, sub(0x4800, mult(0x164A, extract_h(fund_fr_acc))));
            else
                c.dsn_thr = mult(M_fcn, sub(0x3999, mult(0x11D5, extract_h(fund_fr_acc))));
            fund_fr_acc = L_add(fund_fr_acc, fund_fr_step);
        }
        for (ia = index_a; ia < c.index_b; ia++) {
            int iw = window_index(L_tmp);
            tbl[ntbl++] = iw;
            if (iw >= 0 && iw <= 320) {
                int w = WR_SP[iw];
                amp_re = L_mac(amp_re, re[ia], w);
                amp_im = L_mac(amp_im, im[ia], w);
                M_den_sum = L_add(M_den_sum, mult(w, w));
            }
            L_tmp = L_add(L_tmp, 0x400000);
        }
        sc_coef = div_s(0x4000, extract_l(L_shr(M_den_sum, 1)));
        im2 = mult(extract_h(amp_im), sc_coef);
        re2 = mult(extract_h(amp_re), sc_coef);
        for (k = 0, ia = index_a; k < ntbl; k++, ia++) {
            int iw = tbl[k], srr, sri, rt, it;
            if (iw < 0 || iw > 320) {
                srr = sri = 0;
            } else {
                sri = mult(im2, WR_SP[iw]);
                srr = mult(re2, WR_SP[iw]);
            }
            rt = sub(re[ia], srr);
            it = sub(im[ia], sri);
            c.D_num = L_mac(c.D_num, rt, rt);
            c.D_num = L_mac(c.D_num, it, it);
            M_num_sum = L_mac(M_num_sum, re[ia], re[ia]);
            M_num_sum = L_mac(M_num_sum, im[ia], im[ia]);
        }
        c.M_den[j] = sc_coef;
        c.M_num[j] = M_num_sum;
        c.D_den = L_add(c.D_den, M_num_sum);
        num_harms_cnt++;
        if (num_harms_cnt == 3 && c.band_cnt < num_bands - 1) {
            decide(e, &c, j - 2, j + 1);
            c.D_num = c.D_den = 0;
            num_harms_cnt = 0;
            c.band_cnt++;
        }
        acc_a = L_add(acc_a, fund_freq);
        acc_b = L_add(acc_b, fund_freq);
        acc = L_add(acc, fund_freq);
    }
    if (num_harms_cnt)
        decide(e, &c, num_harms - num_harms_cnt, num_harms);

    e->l_uv = c.uv_harms_cnt;
    e->b_vec[1] = c.b1_vec;
    e->b_vec[0] = shr(sub(e->ref_pitch, 0x1380), 7);
}

/* ------------------------------------------------------------------------ */
/*  Spectral amplitudes                                                     */
/* ------------------------------------------------------------------------ */

static void dct(const int *inp, int m_lim, int i_lim, int *out)
{
    int intl, intl2, m, i, begin, step;
    int32_t total;
    if (m_lim == 1) {
        intl = CNST_0_5_Q1_15;
        intl2 = CNST_1_0_Q1_15;
    } else {
        intl = div_s(CNST_0_5_Q5_11, m_lim << 11);
        intl2 = shl(intl, 1);
    }
    intl &= 0xFFFF;
    intl2 &= 0xFFFF;
    total = 0;
    for (m = 0; m < m_lim; m++)
        total = L_add(total, inp[m]);
    out[0] = extract_l(L_mpy_ls(total, s16(intl2)));
    begin = intl;
    step = intl2;
    for (i = 1; i < i_lim; i++) {
        int a = begin;
        total = 0;
        for (m = 0; m < m_lim; m++) {
            total = L_add(total, mult(inp[m], cos_fxp(s16(a))));
            a = (a + step) & 0xFFFF;
        }
        out[i] = extract_l(L_mpy_ls(total, s16(intl2)));
        step = (step + intl2) & 0xFFFF;
        begin = (begin + intl) & 0xFFFF;
    }
}

static void idct(const int *inp, int m_lim, int i_lim, int *out)
{
    int intl, intl2, m, i, step;
    if (m_lim == 1) {
        intl = CNST_0_5_Q1_15;
        intl2 = CNST_1_0_Q1_15;
    } else {
        intl = div_s(CNST_0_5_Q5_11, m_lim << 11);
        intl2 = shl(intl, 1);
    }
    intl &= 0xFFFF;
    intl2 &= 0xFFFF;
    step = intl;
    for (i = 0; i < i_lim; i++) {
        int32_t total = 0;
        int a = step;
        for (m = 1; m < m_lim; m++) {
            total = L_add(total, L_shr(L_mult(inp[m], cos_fxp(s16(a))), 7));
            a = (a + step) & 0xFFFF;
        }
        total = L_add(total, L_shr(L_deposit_h(inp[0]), 8));
        out[i] = extract_l(L_shr_r(total, 8));
        step = (step + intl2) & 0xFFFF;
    }
}

/* Shared interpolation of the previous frame's amplitudes; returns total. */
static int32_t predict(imbe_enc *e, int num_harms, uint32_t k_coef, int ro_coef,
                       const int32_t *values, int is_encode, int32_t *out)
{
    const int32_t *prev = e->sa_prev2;
    uint32_t k_acc = k_coef;
    int32_t total = 0;
    int i;
    for (i = 0; i < num_harms; i++) {
        int index = (int)((k_acc >> 24) & 0xFFFF);
        int si = s16((int64_t)((k_acc - ((uint32_t)index << 24)) & 0xFFFFFFFFu) >> 9);
        if (si == 0) {
            int32_t t = L_mpy_ls(prev[index], ro_coef);
            if (is_encode) {
                out[i] = L_sub(values[i], t);
                total = L_add(total, prev[index]);
            } else {
                out[i] = L_add(values[i], t);
            }
        } else {
            int32_t t = L_mpy_ls(prev[index], sub(0x7FFF, si)), v;
            if (is_encode) {
                total = L_add(total, t);
                v = L_sub(values[i], L_mpy_ls(t, ro_coef));
            } else {
                v = L_add(values[i], L_mpy_ls(t, ro_coef));
            }
            t = L_mpy_ls(prev[index + 1], si);
            if (is_encode) {
                total = L_add(total, t);
                v = L_sub(v, L_mpy_ls(t, ro_coef));
            } else {
                v = L_add(v, L_mpy_ls(t, ro_coef));
            }
            out[i] = v;
        }
        k_acc = k_acc + k_coef;
    }
    return total;
}

static int hi_ord_step(int j, int nb)
{
    return extract_h(s32((int64_t)s32((int64_t)HI_ORD_STD_TBL[j - 1] * HI_ORD_STEP_SIZE_TBL[nb - 1]) << 1)) & 0xFFFF;
}

static void sa_encode(imbe_enc *e)
{
    int num_harms = e->num_harms, prev_n = e->num_harms_prev2;
    int32_t *prev = e->sa_prev2;
    int32_t logs[NUM_HARMS_MAX], vec32[NUM_HARMS_MAX], recon[NUM_HARMS_MAX], total;
    int t_vec[NUM_HARMS_MAX], c_vec[MAX_BLOCK_LEN], gain_vec[NUM_PRED_RES_BLKS], gain_r[NUM_PRED_RES_BLKS];
    const int32_t *gss;
    int *ba = e->bit_alloc, *b_vec = e->b_vec;
    uint32_t k_coef, item;
    int ro_coef, i, j, sh, inv, index, pos, bp, bap;

    if (num_harms == prev_n) {
        k_coef = CNST_ONE_Q8_24;
    } else if (num_harms > prev_n) {
        k_coef = (uint32_t)div_s(prev_n << 9, num_harms << 9) << 9;
    } else {
        int t = prev_n;
        k_coef = 0;
        while (t > num_harms) {
            t -= num_harms;
            k_coef += CNST_ONE_Q8_24;
        }
        k_coef += (uint32_t)div_s(t << 9, num_harms << 9) << 9;
    }

    if (num_harms <= 15)
        ro_coef = CNST_0_4_Q1_15;
    else if (num_harms <= 24)
        ro_coef = s16((int64_t)num_harms * CNST_0_03_Q1_15 - CNST_0_05_Q1_15);
    else
        ro_coef = CNST_0_7_Q1_15;

    for (i = prev_n + 1; i < NUM_HARMS_MAX + 2; i++)
        prev[i] = prev[prev_n];

    for (i = 0; i < num_harms; i++)
        logs[i] = Log2(e->sa[i]);
    total = predict(e, num_harms, k_coef, ro_coef, logs, 1, vec32);

    sh = norm_s(num_harms);
    inv = div_s(0x4000, num_harms << sh);
    total = L_shr(L_mpy_ls(L_mpy_ls(total, ro_coef), inv), 14 - sh);
    for (i = 0; i < num_harms; i++)
        t_vec[i] = extract_h(L_shl(L_add(vec32[i], total), 5));

    index = num_harms - NUM_HARMS_MIN;
    get_bit_allocation(num_harms, ba);
    item = LMPRBL_TBL[index];
    pos = 0;
    bp = 8;
    bap = 5;
    for (i = 0; i < NUM_PRED_RES_BLKS; i++) {
        int bl_len = (int)((item >> 28) & 0xF);
        item <<= 4;
        dct(t_vec + pos, bl_len, bl_len, c_vec);
        gain_vec[i] = c_vec[0];
        for (j = 1; j < bl_len; j++) {
            int nb = ba[bap++];
            if (nb)
                b_vec[bp] = qnt_by_step(c_vec[j], hi_ord_step(j, nb), nb);
            else
                b_vec[bp] = 0;
            bp++;
        }
        pos += bl_len;
    }

    dct(gain_vec, NUM_PRED_RES_BLKS, NUM_PRED_RES_BLKS, gain_r);
    gss = GAIN_STEP_SIZE_TBL + index * 5;
    b_vec[2] = tbl_quant(gain_r[0], GAIN_QNT_TBL, GAIN_QNT_TBL_SIZE);
    for (j = 1; j < 6; j++)
        b_vec[2 + j] = qnt_by_step(gain_r[j], gss[j - 1], ba[j - 1]);

    gain_vec[0] = GAIN_QNT_TBL[b_vec[2]];
    for (i = 1; i < 6; i++)
        gain_vec[i] = extract_l(L_shr(deqnt_by_step(b_vec[2 + i], gss[i - 1], ba[i - 1]), 5));
    idct(gain_vec, NUM_PRED_RES_BLKS, NUM_PRED_RES_BLKS, gain_r);

    memset(t_vec, 0, sizeof(t_vec));
    item = LMPRBL_TBL[index];
    pos = 0;
    bp = 8;
    bap = 5;
    for (i = 0; i < NUM_PRED_RES_BLKS; i++) {
        int bl_len = (int)((item >> 28) & 0xF);
        item <<= 4;
        memset(c_vec, 0, sizeof(c_vec));
        c_vec[0] = gain_r[i];
        for (j = 1; j < bl_len; j++) {
            int nb = ba[bap++];
            if (nb)
                c_vec[j] = extract_l(L_shr(deqnt_by_step(b_vec[bp], hi_ord_step(j, nb), nb), 5));
            else
                c_vec[j] = 0;
            bp++;
        }
        idct(c_vec, bl_len, bl_len, t_vec + pos);
        pos += bl_len;
    }

    for (i = prev_n + 1; i < NUM_HARMS_MAX + 2; i++)
        prev[i] = prev[prev_n];
    for (i = 0; i < num_harms; i++)
        recon[i] = L_shr(L_deposit_h(t_vec[i]), 5);
    predict(e, num_harms, k_coef, ro_coef, recon, 0, vec32);
    for (i = 1; i <= num_harms; i++)
        prev[i] = L_sub(vec32[i - 1], total);
    e->num_harms_prev2 = num_harms;
}

/* ------------------------------------------------------------------------ */
/*  Bit packing                                                             */
/* ------------------------------------------------------------------------ */

static void encode_frame_vector(imbe_enc *e, int *fv)
{
    int num_harms = e->num_harms, num_bands = e->num_bands;
    int *ba = e->bit_alloc, *b_vec = e->b_vec;
    int bits[EN_BIT_STREAM_LEN + 64];
    int idx = 0, bit_thr, bit_mask, i, v, k;

    memset(bits, 0, sizeof(bits));
    get_bit_allocation(num_harms, ba);
    bit_thr = num_harms == 0xB ? 9 : ba[0];
    bit_mask = shl(1, bit_thr - 1);
    while (idx < EN_BIT_STREAM_LEN - (num_bands - 3)) {
        for (i = 0; i < num_harms - 1; i++) {
            if (bit_thr && bit_thr <= ba[i]) {
                bits[idx] = (b_vec[3 + i] & bit_mask) ? 1 : 0;
                idx++;
            }
        }
        bit_thr--;
        bit_mask = shr(bit_mask, 1);
    }

    fv[0] = shl(b_vec[0] & 0xFC, 4) | (b_vec[2] & 0x38);
    fv[0] |= bits[0] ? 4 : 0;
    fv[0] |= bits[1] ? 2 : 0;
    fv[0] |= bits[2] ? 1 : 0;
    idx = 3;
    for (v = 1; v < 4; v++) {
        int t = 0;
        for (k = 0; k < 12; k++)
            t = (t << 1) | bits[idx++];
        fv[v] = s16(t);
    }
    idx -= num_bands + 2;
    bit_mask = shl(1, num_bands - 1);
    for (k = 0; k < num_bands; k++) {
        bits[idx++] = (b_vec[1] & bit_mask) ? 1 : 0;
        bit_mask >>= 1;
    }
    bits[idx++] = (b_vec[2] & 0x04) ? 1 : 0;
    bits[idx++] = (b_vec[2] & 0x02) ? 1 : 0;
    idx -= num_bands + 2;
    for (v = 4; v < 7; v++) {
        int t = 0;
        for (k = 0; k < 11; k++)
            t = (t << 1) | bits[idx++];
        fv[v] = s16(t);
    }
    fv[7] = shl(b_vec[0] & 0x03, 1) | shl(b_vec[2] & 0x01, 3);
    fv[7] |= bits[idx] ? 0x40 : 0;
    fv[7] |= bits[idx + 1] ? 0x20 : 0;
    fv[7] |= bits[idx + 2] ? 0x10 : 0;
    fv[7] |= b_vec[num_harms + 2] ? 0x01 : 0;
}

static void pack_frame_vector(const int *fv, uint8_t *out)
{
    static const int widths[8] = {12, 12, 12, 12, 11, 11, 11, 7};
    uint64_t hi = 0;           /* 88 bits: 64 + 24 */
    uint32_t lo = 0;
    int i, b;
    for (i = 0; i < 8; i++) {
        uint32_t value = (uint32_t)fv[i] & ((1u << widths[i]) - 1);
        for (b = widths[i] - 1; b >= 0; b--) {
            hi = (hi << 1) | (lo >> 23);
            lo = ((lo << 1) | ((value >> b) & 1)) & 0xFFFFFFu;
        }
    }
    for (i = 0; i < 8; i++)
        out[i] = (uint8_t)(hi >> (56 - 8 * i));
    out[8] = (uint8_t)(lo >> 16);
    out[9] = (uint8_t)(lo >> 8);
    out[10] = (uint8_t)lo;
}

/* ------------------------------------------------------------------------ */
/*  Known-tone frames (see imbe.py _apply_tone_hint / _shape_tone_frame)     */
/* ------------------------------------------------------------------------ */

static void apply_tone_hint(imbe_enc *e, int b0)
{
    int ref_pitch = (b0 << 7) + 0x13C0;
    int shift = norm_s(ref_pitch);
    int tmp = div_s(0x4000, shl(ref_pitch, shift));
    e->ref_pitch = ref_pitch;
    e->fund_freq = L_shl(tmp, shift + 11);
}

static void shape_tone_frame(imbe_enc *e, const int *harms, int n_harms)
{
    int n = e->num_harms, is_tone[NUM_HARMS_MAX], any = 0, peak = 0, floor_, j, b;
    memset(is_tone, 0, sizeof(is_tone));
    for (j = 0; j < n_harms; j++) {
        if (harms[j] >= 1 && harms[j] <= n) {
            is_tone[harms[j] - 1] = 1;
            any = 1;
        }
    }
    if (!any)
        return;
    for (j = 0; j < n; j++)
        if (is_tone[j] && e->sa[j] > peak)
            peak = e->sa[j];
    floor_ = peak >> 5;
    if (floor_ < 1)
        floor_ = 1;
    for (j = 0; j < n; j++) {
        if (!is_tone[j])
            e->sa[j] = floor_;
        e->p_v_uv_dsn[j] = 1;
    }
    for (b = 0; b < e->num_bands; b++)
        e->v_uv_dsn[b] = 1;
    e->b_vec[1] = (1 << e->num_bands) - 1;
    e->l_uv = 0;
}

/* ------------------------------------------------------------------------ */
/*  Public API                                                              */
/* ------------------------------------------------------------------------ */

/* the library's API version (encoder and decoder): bump on any change */
EXPORT int imbe_enc_version(void)
{
    return 2;
}

EXPORT void *imbe_enc_new(void)
{
    imbe_enc *e = (imbe_enc *)malloc(sizeof(imbe_enc));
    if (e)
        enc_init(e);
    return e;
}

EXPORT void imbe_enc_free(void *handle)
{
    free(handle);
}

/*
 * Encode 160 samples into an 11-byte IMBE frame.
 *   hint_b0       known-tone pitch index (imbe.py tone_hint), or -1
 *   harms/n_harms the tones' harmonic numbers (1-based) when hinted
 * Returns 0, or -1 where the original encoder would abort (the encoder state
 * is then undefined: create a new one).
 */
EXPORT int imbe_enc_encode(void *handle, const int16_t *samples, int hint_b0,
                           const int *harms, int n_harms, uint8_t *out)
{
    imbe_enc *e = (imbe_enc *)handle;
    jmp_buf jb;
    jmp_buf *saved = g_abort;
    const int keep = PITCH_EST_BUF_SIZE - FRAME;
    int fft_re[FFTLENGTH], re[FFTLENGTH], im[FFTLENGTH], fv[8];
    int i, k, w;

    g_abort = &jb;
    if (setjmp(jb)) {
        g_abort = saved;
        return -1;
    }

    memmove(e->pitch_est_buf, e->pitch_est_buf + FRAME, keep * sizeof(int));
    memmove(e->pitch_ref_buf, e->pitch_ref_buf + FRAME, keep * sizeof(int));
    dc_rmv(e, samples, e->pitch_ref_buf + keep);
    pe_lpf(e, e->pitch_ref_buf + keep, e->pitch_est_buf + keep);
    pitch_est(e, e->pitch_est_buf);
    if (hint_b0 >= 0)
        apply_tone_hint(e, hint_b0);

    memset(fft_re, 0, sizeof(fft_re));
    {
        const int *sig = e->pitch_ref_buf + 40;
        k = 0;
        for (i = 146; i < 256; i++, k++)
            fft_re[i] = mult(sig[k], WR[k]);
        fft_re[0] = sig[k++];
        w = 110 - 1;
        for (i = 1; i < 111; i++, k++, w--)
            fft_re[i] = mult(sig[k], WR[w]);
    }
    fft(e, fft_re, re, im);
    if (hint_b0 < 0)
        pitch_ref(e, re, im);
    v_uv_det(e, re, im);
    if (hint_b0 >= 0)
        shape_tone_frame(e, harms, n_harms);
    sa_encode(e);
    encode_frame_vector(e, fv);
    pack_frame_vector(fv, out);

    g_abort = saved;
    return 0;
}
