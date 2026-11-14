/*
 * IMBE (P25 Phase 1) voice decoder - compiled version.
 *
 * A direct C port of src/p25/mbe_decoder.py (itself a port of mbelib's
 * mbe_decodeImbe4400Parms / mbe_processImbe4400DataF), in double precision
 * with the same evaluation order. The unvoiced synthesis uses random phases
 * and noise; this file has its own random generator, so output matches the
 * Python decoder closely rather than bit for bit (tests/test_imbe_native.py).
 *
 * Build: python tools/build_native.py
 *
 * GPL-2.0 (mbelib is ISC; this project's P25 code is GPL-2.0).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mbe_tables.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define N 160
#define NL 58              /* harmonic slots 0..56, plus mbelib's read one past the end */
#define PI 3.14159265358979323846
#define E_ 2.71828182845904523536

typedef struct {
    double w0;
    int L, K;
    int Vl[NL];
    double Ml[NL], log2Ml[NL], PHIl[NL], PSIl[NL];
    int repeat;
} parms;

typedef struct {
    int uvquality;
    uint64_t rng;
    parms cur, prev, prev_enh;
} imbe_dec;

/* splitmix64: plenty for synthesis noise */
static double rnd(imbe_dec *d)
{
    uint64_t z = (d->rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

static double rand_phase(imbe_dec *d)
{
    return rnd(d) * (2 * PI) - PI;
}

static void parms_init(parms *p)
{
    int l;
    memset(p, 0, sizeof(*p));
    p->w0 = 0.09378;
    p->L = 30;
    p->K = 10;
    for (l = 0; l < NL; l++)
        p->PSIl[l] = PI / 2;
}

static void parms_copy(parms *dst, const parms *src)
{
    *dst = *src;
    dst->Ml[0] = 0.0;
}

static void init_parms(imbe_dec *d)
{
    parms_init(&d->prev);
    parms_init(&d->cur);
    parms_copy(&d->cur, &d->prev);
    parms_init(&d->prev_enh);
    parms_copy(&d->prev_enh, &d->prev);
}

/* ------------------------------------------------------------------------ */
/*  Parameter decoding (mbe_decodeImbe4400Parms)                             */
/* ------------------------------------------------------------------------ */

static int decode_params(imbe_dec *d, const uint8_t *frame)
{
    parms *cur = &d->cur, *prev = &d->prev;
    int bits[88], bb[58][12];
    double Gm[7], Ri[7], Cik[7][11], Tl[57], deltal[57], s77, rho;
    int intkl[57];
    int i, j, k, l, m, b0, L, L9, K, b2;

    for (i = 0; i < 88; i++)
        bits[i] = (frame[i >> 3] >> (7 - (i & 7))) & 1;
    cur->repeat = prev->repeat;
    b0 = 0;
    for (i = 0; i < 6; i++)
        b0 = (b0 << 1) | bits[i];
    b0 = (b0 << 1) | bits[85];
    b0 = (b0 << 1) | bits[86];
    if (b0 > 207)
        return 0;
    cur->w0 = (4 * PI) / (b0 + 39.5);
    L = (int)(0.9254 * (int)((PI / cur->w0) + 0.25));
    if (L > 56 || L < 9)
        return 0;
    cur->L = L;
    L9 = L - 9;
    K = L < 37 ? (int)((L + 2) / 3.0) : 12;
    cur->K = K;

    memset(bb, 0, sizeof(bb));
    for (i = 6; i < 85; i++)
        bb[BO[L9][i - 6][0]][BO[L9][i - 6][1]] = bits[i];

    j = 1;
    k = K - 1;
    for (i = 1; i <= L; i++) {
        cur->Vl[i] = bb[1][k];
        if (j == 3) {
            j = 1;
            k = k > 0 ? k - 1 : 0;
        } else {
            j++;
        }
    }

    b2 = 0;
    for (i = 5; i >= 0; i--)
        b2 = (b2 << 1) | bb[2][i];
    memset(Gm, 0, sizeof(Gm));
    Gm[1] = B2[b2];
    for (i = 2; i < 7; i++) {
        int nbits = BA_BITS[L9][i - 2], bm = 0, jj;
        double step = BA_STEP[L9][i - 2];
        for (jj = nbits - 1; jj >= 0; jj--)
            bm = (bm << 1) | bb[i + 1][jj];
        if (nbits >= 1)
            Gm[i] = step * ((double)(bm - (1 << (nbits - 1))) + 0.5);
        else
            Gm[i] = step * (((double)bm - 0.5) + 0.5);
    }

    for (i = 1; i < 7; i++) {
        double s = 0.0;
        for (m = 1; m < 7; m++) {
            int am = m == 1 ? 1 : 2;
            s += am * Gm[m] * cos(PI * (m - 1) * (i - 0.5) / 6);
        }
        Ri[i] = s;
    }

    memset(Cik, 0, sizeof(Cik));
    m = 8;
    for (i = 1; i < 7; i++) {
        Cik[i][1] = Ri[i];
        for (k = 2; k <= IMBEJI[L9][i - 1]; k++) {
            int Bm = HOBA[L9][m - 8];
            if (Bm == 0) {
                Cik[i][k] = 0.0;
            } else {
                int bm = 0, b;
                for (b = 0; b < Bm; b++)
                    bm = (bm << 1) | bb[m][Bm - b - 1];
                Cik[i][k] = (QUANTSTEP[Bm - 1] * STANDDEV[k - 2]) * ((double)(bm - (1 << (Bm - 1))) + 0.5);
            }
            m++;
        }
    }

    memset(Tl, 0, sizeof(Tl));
    l = 1;
    for (i = 1; i < 7; i++) {
        int ji = IMBEJI[L9][i - 1], jj;
        for (jj = 1; jj <= ji; jj++) {
            double s = 0.0;
            for (k = 1; k <= ji; k++) {
                int ak = k == 1 ? 1 : 2;
                s += ak * Cik[i][k] * cos(PI * (k - 1) * (jj - 0.5) / ji);
            }
            Tl[l++] = s;
        }
    }

    if (L <= 15)
        rho = 0.4;
    else if (L <= 24)
        rho = 0.03 * L - 0.05;
    else
        rho = 0.7;
    if (L > prev->L) {
        for (l = prev->L + 1; l <= L; l++) {
            prev->Ml[l] = prev->Ml[prev->L];
            prev->log2Ml[l] = prev->log2Ml[prev->L];
        }
    }
    /* mbelib reads log2Ml[intkl + 1], one past the end when the previous
       frame had 56 harmonics; slot 57 is always 0 and its weight is 0. */
    prev->log2Ml[57] = 0.0;
    s77 = 0.0;
    for (l = 1; l <= L; l++) {
        double fl = ((double)prev->L / L) * l;
        intkl[l] = (int)fl;
        deltal[l] = fl - intkl[l];
        s77 += (1 - deltal[l]) * prev->log2Ml[intkl[l]] + deltal[l] * prev->log2Ml[intkl[l] + 1];
    }
    s77 = (rho / L) * s77;
    for (l = 1; l <= L; l++) {
        double c1 = rho * (1 - deltal[l]) * prev->log2Ml[intkl[l]];
        double c2 = rho * deltal[l] * prev->log2Ml[intkl[l] + 1];
        cur->log2Ml[l] = Tl[l] + c1 + c2 - s77;
        cur->Ml[l] = pow(2.0, cur->log2Ml[l]);
    }
    return 1;
}

/* ------------------------------------------------------------------------ */
/*  Synthesis (mbe_processImbe4400DataF)                                     */
/* ------------------------------------------------------------------------ */

static void enhance(parms *mp)
{
    int L = mp->L, l;
    double *Ml = mp->Ml;
    double Rm0 = 0, Rm1 = 0, R2m0, R2m1, total, gamma;
    for (l = 1; l <= L; l++)
        Rm0 += Ml[l] * Ml[l];
    for (l = 1; l <= L; l++)
        Rm1 += Ml[l] * Ml[l] * cos(mp->w0 * l);
    R2m0 = Rm0 * Rm0;
    R2m1 = Rm1 * Rm1;
    for (l = 1; l <= L; l++) {
        if (Ml[l] != 0) {
            double den = mp->w0 * Rm0 * (R2m0 - R2m1);
            double num = 0.96 * PI * ((R2m0 + R2m1) - 2 * Rm0 * Rm1 * cos(mp->w0 * l));
            double ratio = den != 0 ? num / den : 0.0;
            double Wl = sqrt(Ml[l]) * (ratio > 0 ? pow(ratio, 0.25) : 0.0);
            if (8 * l <= L)
                ;
            else if (Wl > 1.2)
                Ml[l] *= 1.2;
            else if (Wl < 0.5)
                Ml[l] *= 0.5;
            else
                Ml[l] *= Wl;
        }
    }
    total = 0;
    for (l = 1; l <= L; l++)
        total += Ml[l] * Ml[l];
    gamma = total == 0 ? 1.0 : sqrt(Rm0 / total);
    for (l = 1; l <= L; l++)
        Ml[l] *= gamma;
}

typedef struct {
    int q;
    double uvthreshold, uvsine, uvrand, qfactor, uvstep, uvoffset;
} uvcfg;

/* adds weight * unvoiced(w0, l) to out */
static void add_unvoiced(imbe_dec *d, const uvcfg *c, double w0, int l, const double *phases,
                         const double *ws, double Ml, double *out)
{
    double buf[N];
    double w0l = w0 * l;
    int i, k;
    memset(buf, 0, sizeof(buf));
    for (i = 0; i < c->q; i++) {
        double mul = l + i * c->uvstep - c->uvoffset;
        for (k = 0; k < N; k++)
            buf[k] += cos(w0 * k * mul + phases[i]);
        if (w0l > c->uvthreshold) {
            double amp = (w0l - c->uvthreshold) * c->uvrand;
            for (k = 0; k < N; k++)
                buf[k] += amp * rnd(d);
        }
    }
    for (k = 0; k < N; k++)
        out[k] += buf[k] * c->uvsine * c->qfactor * ws[k] * Ml;
}

static void synthesize(imbe_dec *d, parms *cur, parms *prev, double *out)
{
    uvcfg c;
    const double *ws_cur = WS, *ws_prev = WS + N;
    double cw0 = cur->w0, pw0 = prev->w0, ph[8], ph2[8];
    int num_uv = 0, maxl, l, k, i;

    c.q = d->uvquality;
    c.uvthreshold = (2700.0 * PI) / 4000;
    c.uvsine = 1.3591409 * E_;
    c.uvrand = 2.0;
    c.qfactor = c.q == 1 ? (1 / E_) : log((double)c.q) / c.q;
    c.uvstep = 1.0 / c.q;
    c.uvoffset = (c.uvstep * (c.q - 1)) / 2;
    for (l = 1; l <= cur->L; l++)
        if (cur->Vl[l] == 0)
            num_uv++;
    memset(out, 0, N * sizeof(double));

    if (cur->L > prev->L) {
        maxl = cur->L;
        for (l = prev->L + 1; l <= maxl; l++) {
            prev->Ml[l] = 0.0;
            prev->Vl[l] = 1;
        }
    } else {
        maxl = prev->L;
        for (l = cur->L + 1; l <= maxl; l++) {
            cur->Ml[l] = 0.0;
            cur->Vl[l] = 1;
        }
    }

    for (l = 1; l < 57; l++) {
        cur->PSIl[l] = prev->PSIl[l] + (pw0 + cw0) * (l * N / 2.0);
        if (l <= cur->L / 4)
            cur->PHIl[l] = cur->PSIl[l];
        else
            cur->PHIl[l] = cur->PSIl[l] + (num_uv * rand_phase(d)) / cur->L;
    }

    for (l = 1; l <= maxl; l++) {
        double cw0l = cw0 * l, pw0l = pw0 * l;
        int cv = cur->Vl[l], pv = prev->Vl[l];
        if (cv == 0 && pv == 1) {
            for (i = 0; i < c.q; i++)
                ph[i] = rand_phase(d);
            for (k = 0; k < N; k++)
                out[k] += ws_prev[k] * prev->Ml[l] * cos(pw0l * k + prev->PHIl[l]);
            add_unvoiced(d, &c, cw0, l, ph, ws_cur, cur->Ml[l], out);
        } else if (cv == 1 && pv == 0) {
            for (i = 0; i < c.q; i++)
                ph[i] = rand_phase(d);
            for (k = 0; k < N; k++)
                out[k] += ws_cur[k] * cur->Ml[l] * cos(cw0l * (k - N) + cur->PHIl[l]);
            add_unvoiced(d, &c, pw0, l, ph, ws_prev, prev->Ml[l], out);
        } else if (cv == 1 || pv == 1) {
            for (k = 0; k < N; k++) {
                out[k] += ws_prev[k] * prev->Ml[l] * cos(pw0l * k + prev->PHIl[l]);
                out[k] += ws_cur[k] * cur->Ml[l] * cos(cw0l * (k - N) + cur->PHIl[l]);
            }
        } else {
            for (i = 0; i < c.q; i++)
                ph[i] = rand_phase(d);
            for (i = 0; i < c.q; i++)
                ph2[i] = rand_phase(d);
            add_unvoiced(d, &c, pw0, l, ph, ws_prev, prev->Ml[l], out);
            add_unvoiced(d, &c, cw0, l, ph2, ws_cur, cur->Ml[l], out);
        }
    }
}

/* ------------------------------------------------------------------------ */
/*  Public API                                                              */
/* ------------------------------------------------------------------------ */

EXPORT void *imbe_dec_new(int uvquality, uint64_t seed)
{
    imbe_dec *d = (imbe_dec *)malloc(sizeof(imbe_dec));
    if (!d)
        return NULL;
    d->uvquality = uvquality < 1 ? 1 : (uvquality > 8 ? 8 : uvquality);
    d->rng = seed;
    init_parms(d);
    return d;
}

EXPORT void imbe_dec_free(void *handle)
{
    free(handle);
}

/* One 11-byte frame -> 160 samples (mbelib scale). Returns 1 if the frame
   was valid, 0 if it was repeated/muted instead. */
EXPORT int imbe_dec_decode(void *handle, const uint8_t *frame, double *out)
{
    imbe_dec *d = (imbe_dec *)handle;
    int valid = decode_params(d, frame);
    if (!valid) {
        parms_copy(&d->cur, &d->prev);
        d->cur.repeat++;
    } else {
        d->cur.repeat = 0;
    }
    if (d->cur.repeat <= 3) {
        parms_copy(&d->prev, &d->cur);
        enhance(&d->cur);
        synthesize(d, &d->cur, &d->prev_enh, out);
        parms_copy(&d->prev_enh, &d->cur);
        return valid;
    }
    init_parms(d);
    memset(out, 0, N * sizeof(double));
    return valid;
}
