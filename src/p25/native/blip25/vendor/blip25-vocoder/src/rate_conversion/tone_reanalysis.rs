//! Magnitude envelope for a bridged tone frame, measured by the engine's own
//! amplitude-estimation stage.
//!
//! ## What is broken without this
//!
//! §2.10.3 Eq. 206–209 describes a half-rate tone as an MBE frame whose energy
//! sits on harmonics `l₁` and `l₂` and *nowhere else*: every other `M̃_l` is
//! exactly zero. That is a faithful reading of the equations and it is fine for
//! the half-rate synthesiser, which consumes the amplitudes directly. It is not
//! survivable for the full-rate quantiser, which works in `log₂ M̃` — a zero
//! amplitude is `−∞` there, and one `−∞` contaminates every coefficient of the
//! DCT block it lands in, which pins the transmitted gain to the bottom of the
//! Annex E codebook. A DTMF digit converted half → full comes out ~60 dB down:
//! inaudible.
//!
//! ## What replaces it
//!
//! A real analyser never reports a zero: a sinusoid seen through a finite
//! window has skirts, and those skirts are what the codebook floor is standing
//! in for. So ask the encoder. [`crate::engine::SpectrumProbe`] runs the
//! reference front end — DC blocker, 383-sample ring, 199-tap Hamming window,
//! 256-point transform — over a synthesised copy of the tone and reports the
//! harmonic log magnitudes the packer's own amplitude stage would have
//! integrated, sampled on the *bridge's* fundamental rather than on a pitch the
//! encoder picked for itself.
//!
//! Only the magnitude envelope comes from that measurement. `ω̃₀`, `L` and the
//! per-harmonic voicing stay exactly as Eq. 206–208 produced them.
//!
//! ## The scale bridge, and why it carries no fitted constant
//!
//! The probe reports base-2 logarithms at Q11 ([`crate::engine::LOG_MAG_Q_SHIFT`]),
//! so a word `w` is the linear magnitude `2^(w / 2048)`. That exponent is not
//! chosen here, and it is not swept: the engine's own log→linear expansion
//! evaluates `2^(w · 2^(4−15))` and derives its block exponent as `w >> 11`,
//! and the 6-bit gain codebook that gets added straight onto the same vector is
//! the published Annex E gain table times 2048 —
//! `tone_reanalysis::tests::q11_scale_matches_the_engine_gain_codebook` checks
//! that entry by entry against [`crate::fullrate::dequantize::decode_gain`].
//!
//! The *level* needs no constant at all. Band integration is homogeneous in the
//! input scale — scaling the synthesised tone by `g` adds `2048·log₂ g` to every
//! word — so the synthesis amplitude is arbitrary and the envelope is anchored
//! afterwards: the harmonic at `l₁` is put back to exactly the Eq. 209
//! amplitude the tone frame asked for, and every other harmonic follows it. The
//! bridge's level intent is preserved bit for bit; only the shape around it
//! comes from the measurement.
//!
//! ## Cost
//!
//! No look-ahead: the tone is synthesised from the frame's own `(I_D, A_D)`, so
//! the converter's delay is unchanged. The measured envelope depends only on
//! `I_D` — the amplitude anchor is applied after the fact — so a run of tone
//! frames on one ID measures once and reuses, and [`ToneEnvelopes`] remembers
//! the last few IDs so a signalling *cadence* that cycles between tones does
//! not re-measure on every switch.

use core::f32::consts::PI;

use crate::engine::{Mode, SpectrumProbe};

use crate::halfrate::frame::{ToneParams, ANNEX_T};
use crate::mbe_params::{MbeParams, L_MAX};

/// Peak amplitude, in PCM counts, of each component of the synthesised probe
/// signal.
///
/// Arbitrary by construction — the level is anchored to Eq. 209 afterwards, so
/// this only has to be large enough to keep the analyser's skirts off its
/// `−0x7800` "band normalised to zero" floor and small enough that a two-
/// component tone cannot clip. Half of full scale satisfies both.
const PROBE_PEAK: f64 = 16384.0;

/// The harmonic-spacing step [`SpectrumProbe::harmonic_log_mags`] wants, as a
/// Q11 increment over the 129-bin spectrum, for a fundamental of `omega_0`
/// radians per sample.
///
/// One bin of the front end's 256-point transform of 8 kHz audio is 31.25 Hz,
/// so a fundamental of `ω₀` sits `ω₀ · 128/π` bins up.
fn step_q11(omega_0: f32) -> i16 {
    let bins = f64::from(omega_0) * 128.0 / f64::from(PI);
    let q = (bins * 2048.0).round();
    q.clamp(1.0, f64::from(i16::MAX)) as i16
}

/// Render `SpectrumProbe::PRIMING_HALF_FRAMES` half-frames of the Annex T tone
/// `row` describes and push them all through a fresh probe.
///
/// The tone's two components are harmonics `l₁` and `l₂` of `f₀`, i.e.
/// `l₁·f₀` and `l₂·f₀` Hz; a single-frequency tone has `l₁ == l₂` and is
/// rendered once, not twice.
fn probe_of(row: ToneParams) -> SpectrumProbe {
    let ToneParams { f0, l1, l2 } = row;
    let f0 = f64::from(f0);
    let mut freqs = [f0 * f64::from(l1), f0 * f64::from(l2)];
    let n_comp = if l1 == l2 { 1 } else { 2 };
    for f in freqs.iter_mut() {
        *f *= 2.0 * f64::from(PI) / 8000.0;
    }

    let mut probe = SpectrumProbe::new(Mode::FullRate);
    let mut pcm = [0i16; SpectrumProbe::HALF_FRAME_SAMPLES];
    let mut n = 0u32;
    for _ in 0..SpectrumProbe::PRIMING_HALF_FRAMES {
        for s in pcm.iter_mut() {
            let t = f64::from(n);
            let mut v = 0.0f64;
            for &w in &freqs[..n_comp] {
                v += PROBE_PEAK * (w * t).sin();
            }
            *s = v.round().clamp(-32768.0, 32767.0) as i16;
            n += 1;
        }
        probe.push_half_frame(&pcm);
    }
    probe
}

/// The measured envelope of one Annex T tone, in the probe's Q11 log domain
/// and on that tone's own `(ω̃₀, L)` grid.
///
/// Cached by `I_D` in [`ToneEnvelopes`]: the words depend on the tone identity
/// alone, because the Eq. 209 amplitude is applied as a level anchor after the
/// fact.
#[derive(Clone, Debug)]
struct Envelope {
    /// Harmonic count the words were measured for.
    l: u8,
    /// Q11 base-2 log magnitudes, entries `0..l`.
    log_q11: [i16; L_MAX as usize],
}

/// One remembered measurement, with the `I_D` it was measured for. `Envelope`
/// carries the other half of the key, its `L`.
#[derive(Clone, Debug)]
struct Entry {
    id: u8,
    env: Envelope,
}

/// How many distinct tone identities are remembered at once.
///
/// A run of frames on one tone is served by any size at all, so the number is
/// set by *alternation*, and the reference corpus's alternation is a signalling
/// cadence rather than a dual-tone digit: a DTMF digit is a single Annex T row
/// with `l₁ ≠ l₂` and therefore one envelope, and every pure DTMF stem in
/// `tv-rc` already hits a one-entry cache on every frame after its first per
/// digit. What thrashes is `alert`, which cycles three tones on a 17-frame
/// cadence, and `xfer`, which alternates two. Simulating an LRU of each size
/// over the tone-identity sequence of all 58 tone-bearing stems, three entries
/// leaves nothing but compulsory misses everywhere and four through eight
/// remove not one further miss:
///
/// ```text
///   stem      tone frames   distinct   misses at k = 1 / 2 / 3 / 4 … 8
///   alert            153          3           9 /  9 /  3 /  3 …  3
///   xfer             106          2           6 /  2 /  2 /  2 …  2
///   dtmf            1192         20          63 / 53 / 38 / 38 … 38
///   dtmf180         2883         16          16 / 16 / 16 / 16 … 16
/// ```
const REMEMBERED: usize = 3;

/// The last few tone envelopes, remembered.
///
/// Measuring one envelope runs the reference front end over six half-frames of
/// synthesised tone and costs ~0.31 ms; a hit costs one integer comparison per
/// live entry. Entries are held most-recently-used first, so a cadence shorter
/// than [`REMEMBERED`] measures each of its tones once however long it cycles.
///
/// This is a pure cache. The measurement is a function of `I_D` alone — `ω̃₀`
/// and `L` are derived from it by Eq. 206–207, and the Eq. 209 level is
/// anchored afterwards — so evicting an entry can only cost time, never change
/// a bit of output.
#[derive(Clone, Debug, Default)]
pub(crate) struct ToneEnvelopes {
    /// Most-recently-used first; `None` slots are free and sit at the back.
    entries: [Option<Entry>; REMEMBERED],
    /// How many envelopes have actually been measured. Test-only: it is the
    /// only way to tell a hit from a miss from outside, since both return the
    /// same words.
    #[cfg(test)]
    measurements: u32,
}

impl ToneEnvelopes {
    /// Replace `params`' magnitude envelope with the measured one, keeping its
    /// `ω̃₀`, `L` and voicing, and anchoring the level so that harmonic `l₁`
    /// lands on exactly the Eq. 209 amplitude `params` already carries.
    ///
    /// Returns `None` — leaving the caller with the Eq. 206–209 parameters
    /// unchanged — when there is nothing to anchor to: a reserved `I_D`, the
    /// `I_D = 255` silence sentinel whose Eq. 209 amplitude is deliberately
    /// zero, or an `l₁` outside `[1, L]`.
    pub(crate) fn substitute_magnitudes(
        &mut self,
        id: u8,
        params: &MbeParams,
    ) -> Option<MbeParams> {
        let row = ANNEX_T[id as usize]?;
        let l = params.harmonic_count();
        let anchor = usize::from(row.l1).checked_sub(1)?;
        if anchor >= usize::from(l) {
            return None;
        }
        // The Eq. 209 amplitude this frame asked for. Zero (the silence
        // sentinel) has no log and nothing to anchor.
        let target_log2 = params.amplitudes_slice()[anchor].log2();
        if !target_log2.is_finite() {
            return None;
        }

        let env = self.envelope(id, params.omega_0(), l);
        let measured = f32::from(env.log_q11[anchor]) / 2048.0;
        let shift = target_log2 - measured;

        let mut amps = [0f32; L_MAX as usize];
        for (a, &w) in amps[..usize::from(l)]
            .iter_mut()
            .zip(env.log_q11[..usize::from(l)].iter())
        {
            *a = (f32::from(w) / 2048.0 + shift).exp2();
        }
        MbeParams::new(
            params.omega_0(),
            l,
            params.voiced_slice(),
            &amps[..usize::from(l)],
        )
        .ok()
    }

    /// The remembered envelope for `id`, measuring it if this tone is not held.
    ///
    /// Either way the answer ends up in slot 0, so the entries stay ordered
    /// most-recently-used first and the entry evicted on a miss is the one
    /// unused for longest.
    fn envelope(&mut self, id: u8, omega_0: f32, l: u8) -> &Envelope {
        match self
            .entries
            .iter()
            .position(|e| matches!(e, Some(e) if e.id == id && e.env.l == l))
        {
            // Already the most recent: nothing to reorder.
            Some(0) => {}
            // Promote: slot `i` to the front, everything above it down one.
            Some(i) => self.entries[..=i].rotate_right(1),
            None => {
                let row = ANNEX_T[id as usize].expect("caller checked the Annex T row");
                let probe = probe_of(row);
                let mut log_q11 = [0i16; L_MAX as usize];
                probe.harmonic_log_mags(step_q11(omega_0), &mut log_q11[..usize::from(l)]);
                #[cfg(test)]
                {
                    self.measurements += 1;
                }
                // Rotating the whole array brings the least-recently-used entry
                // (or a free slot) to the front, where it is overwritten.
                self.entries.rotate_right(1);
                self.entries[0] = Some(Entry {
                    id,
                    env: Envelope { l, log_q11 },
                });
            }
        }
        &self.entries[0]
            .as_ref()
            .expect("just promoted or populated")
            .env
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LOG_MAG_Q_SHIFT;
    use crate::fullrate::dequantize::decode_gain;
    use crate::halfrate::dequantize::tone_to_mbe_params;

    /// The one scale constant in this module, checked against the engine rather
    /// than chosen. `GAIN64` is the 6-bit gain codebook the engine adds
    /// straight onto its Q11 log-magnitude vector, and Annex E is the same
    /// codebook as the published spec states it; if the vector is Q11 then one
    /// is 2048 times the other.
    ///
    /// Entries 0..8 are excluded: the engine extends the codebook's low end
    /// with eight extra steps of −0.25 log₂ below Annex E's bottom entry, which
    /// says nothing about the scale.
    #[test]
    fn q11_scale_matches_the_engine_gain_codebook() {
        assert_eq!(LOG_MAG_Q_SHIFT, 11);
        let scale = f32::from(1u16 << LOG_MAG_Q_SHIFT);
        assert_eq!(scale, 2048.0);
        for idx in 8..64u8 {
            let engine = f32::from(crate::engine::tables::GAIN64[usize::from(idx)]) / scale;
            let spec = decode_gain(idx);
            assert!(
                (engine - spec).abs() < 5e-4,
                "gain index {idx}: engine {engine} vs Annex E {spec} — the log-magnitude \
                 vector is not Q11"
            );
        }
    }

    /// The step derivation, checked against the analyser instead of asserted:
    /// a tone at `l₁·f₀` measured on a grid of spacing `ω̃₀ = 2π f₀/8000` must
    /// peak at harmonic `l₁`.
    #[test]
    fn the_measured_envelope_peaks_on_the_tone_harmonic() {
        let mut checked = 0;
        for id in 0..=254u8 {
            let Some(row) = ANNEX_T[usize::from(id)] else {
                continue;
            };
            // Amplitude 127 is Eq. 209's full scale; the peak location does
            // not depend on it.
            let Some(params) = tone_to_mbe_params(id, 127) else {
                continue;
            };
            let l = params.harmonic_count();
            if row.l1 == 0 || usize::from(row.l1) > usize::from(l) {
                continue;
            }
            let probe = probe_of(row);
            let mut log_q11 = [0i16; L_MAX as usize];
            probe.harmonic_log_mags(step_q11(params.omega_0()), &mut log_q11[..usize::from(l)]);
            let (peak, _) = log_q11[..usize::from(l)]
                .iter()
                .enumerate()
                .max_by_key(|(_, &w)| w)
                .expect("l >= 9");
            let want = [usize::from(row.l1) - 1, usize::from(row.l2) - 1];
            assert!(
                want.contains(&peak),
                "tone {id} (f0 {} Hz, l1 {} l2 {}): envelope peaks at harmonic {} not {want:?}",
                row.f0,
                row.l1,
                row.l2,
                peak + 1
            );
            checked += 1;
        }
        assert!(checked > 100, "only {checked} Annex T rows exercised");
    }

    /// Anchoring is exact: the substituted envelope reproduces the Eq. 209
    /// amplitude at `l₁` to the last bit of `f32` round-tripping, whatever the
    /// tone and whatever the level.
    #[test]
    fn the_anchor_reproduces_the_eq_209_amplitude() {
        let mut envs = ToneEnvelopes::default();
        let mut checked = 0;
        for id in [5u8, 8, 17, 40, 96, 128, 150] {
            for amplitude in [0u8, 40, 96, 127] {
                let Some(params) = tone_to_mbe_params(id, amplitude) else {
                    continue;
                };
                let row = ANNEX_T[usize::from(id)].expect("row");
                let anchor = usize::from(row.l1) - 1;
                let Some(out) = envs.substitute_magnitudes(id, &params) else {
                    continue;
                };
                let want = params.amplitudes_slice()[anchor];
                let got = out.amplitudes_slice()[anchor];
                assert!(
                    (got - want).abs() <= want * 1e-4,
                    "tone {id} A_D {amplitude}: anchored to {got}, Eq. 209 wants {want}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 20, "only {checked} (id, A_D) pairs exercised");
    }

    /// The point of the whole module: no harmonic comes back at zero, so
    /// nothing downstream sees `log₂ 0`.
    #[test]
    fn no_harmonic_is_left_at_zero() {
        let mut envs = ToneEnvelopes::default();
        for id in 0..=254u8 {
            let Some(params) = tone_to_mbe_params(id, 100) else {
                continue;
            };
            let Some(out) = envs.substitute_magnitudes(id, &params) else {
                continue;
            };
            let l = usize::from(out.harmonic_count());
            let zeros = out.amplitudes_slice()[..l]
                .iter()
                .filter(|m| **m <= 0.0)
                .count();
            assert_eq!(zeros, 0, "tone {id}: {zeros} of {l} harmonics still zero");
            // Eq. 206–209 leaves all but one or two at zero; the whole reason
            // this module exists is that it must not.
            let bridge_zeros = params.amplitudes_slice()[..l]
                .iter()
                .filter(|m| **m <= 0.0)
                .count();
            assert!(
                bridge_zeros >= l - 2,
                "tone {id}: the Eq. 206–209 bridge left only {bridge_zeros} zeros, so this \
                 test is no longer measuring the substitution"
            );
        }
    }

    /// Pitch, harmonic count and voicing are the bridge's, untouched. If a
    /// future edit lets the measurement reach any of them, this fails.
    #[test]
    fn only_the_magnitudes_are_substituted() {
        let mut envs = ToneEnvelopes::default();
        for id in [5u8, 17, 40, 96, 150] {
            let params = tone_to_mbe_params(id, 110).expect("params");
            let out = envs
                .substitute_magnitudes(id, &params)
                .expect("substituted");
            assert_eq!(out.omega_0(), params.omega_0(), "tone {id}: ω̃₀ moved");
            assert_eq!(
                out.harmonic_count(),
                params.harmonic_count(),
                "tone {id}: L moved"
            );
            assert_eq!(
                out.voiced_slice(),
                params.voiced_slice(),
                "tone {id}: voicing moved"
            );
        }
    }

    /// The cache must not be able to answer with another tone's envelope.
    #[test]
    fn the_cache_is_keyed_on_the_tone_identity() {
        let mut cached = ToneEnvelopes::default();
        let ids = [5u8, 96, 5, 150, 5];
        let mut got = Vec::new();
        for id in ids {
            let params = tone_to_mbe_params(id, 100).expect("params");
            got.push(
                cached
                    .substitute_magnitudes(id, &params)
                    .expect("substituted"),
            );
        }
        // Same ID before and after two other tones → identical magnitudes.
        assert_eq!(
            got[0].amplitudes_slice(),
            got[2].amplitudes_slice(),
            "revisiting tone 5 gave a different envelope"
        );
        assert_eq!(got[0].amplitudes_slice(), got[4].amplitudes_slice());
        // Different IDs → different magnitudes (otherwise the key is dead).
        assert_ne!(
            got[0].amplitudes_slice(),
            got[1].amplitudes_slice(),
            "tone 96 got tone 5's envelope"
        );
        assert_ne!(got[1].amplitudes_slice(), got[3].amplitudes_slice());
    }

    /// The whole point of remembering more than one tone: a signalling cadence
    /// that cycles between tones must measure each of them **once**, however
    /// long it cycles.
    ///
    /// The pattern is `alert`'s, the worst alternation in the reference corpus:
    /// three tones, 17 frames each, repeated three times. A one-entry cache
    /// measures 9 times because every switch evicts the tone that is about to
    /// come back.
    #[test]
    fn a_cycling_cadence_measures_each_tone_once() {
        // Three distinct Annex T identities, as `alert` cycles them.
        let cadence = [30u8, 46, 57];
        assert!(
            cadence.len() <= REMEMBERED,
            "the cadence must fit, or this is measuring capacity instead of recency"
        );
        let mut envs = ToneEnvelopes::default();
        for _repeat in 0..3 {
            for id in cadence {
                for _hold in 0..17 {
                    let params = tone_to_mbe_params(id, 100).expect("params");
                    envs.substitute_magnitudes(id, &params)
                        .expect("substituted");
                }
            }
        }
        assert_eq!(
            envs.measurements,
            cadence.len() as u32,
            "153 frames cycling {} tones cost {} measurements; a cache that holds \
             the cadence costs {}",
            cadence.len(),
            envs.measurements,
            cadence.len()
        );
    }

    /// One entry past capacity and recency has to bite: a four-tone cycle
    /// cannot be held by a three-entry cache, and must fall back to measuring
    /// on every switch. This pins [`REMEMBERED`] from the other side, so the
    /// test above cannot be satisfied by an unbounded cache.
    #[test]
    fn a_cadence_one_longer_than_the_cache_still_thrashes() {
        let mut cadence: Vec<u8> = Vec::new();
        for id in 0..=254u8 {
            if ANNEX_T[usize::from(id)].is_some() && tone_to_mbe_params(id, 100).is_some() {
                cadence.push(id);
            }
            if cadence.len() > REMEMBERED {
                break;
            }
        }
        assert_eq!(cadence.len(), REMEMBERED + 1);
        let repeats = 4u32;
        let mut envs = ToneEnvelopes::default();
        for _ in 0..repeats {
            for &id in &cadence {
                let params = tone_to_mbe_params(id, 100).expect("params");
                envs.substitute_magnitudes(id, &params)
                    .expect("substituted");
            }
        }
        assert_eq!(
            envs.measurements,
            repeats * cadence.len() as u32,
            "a {}-tone cycle fits a {REMEMBERED}-entry cache — REMEMBERED is not what \
             this module says it is",
            cadence.len()
        );
    }

    /// A cache may not change a single bit. Whatever the access pattern —
    /// hits, promotions, evictions — every substituted frame must equal what a
    /// converter with no memory at all would have produced.
    #[test]
    fn remembering_changes_no_output_bit() {
        // Enough distinct tones to overflow the cache several times over, in an
        // order that mixes long holds, tight alternation and revisits after
        // eviction.
        let ids: Vec<u8> = (0..=254u8)
            .filter(|&id| tone_to_mbe_params(id, 100).is_some())
            .take(4 * REMEMBERED)
            .collect();
        assert!(
            ids.len() >= 4 * REMEMBERED,
            "not enough usable Annex T rows"
        );

        let mut order: Vec<u8> = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            order.extend(core::iter::repeat_n(id, 3));
            order.push(ids[i / 2]);
            order.push(id);
            order.push(ids[0]);
        }

        let mut shared = ToneEnvelopes::default();
        let mut evictions = 0usize;
        for (i, &id) in order.iter().enumerate() {
            let amplitude = (40 + (i as u8 % 88)).min(127);
            let params = tone_to_mbe_params(id, amplitude).expect("params");
            let before = shared.measurements;
            let was_full = shared.entries[REMEMBERED - 1].is_some();
            let got = shared
                .substitute_magnitudes(id, &params)
                .expect("substituted");
            if shared.measurements != before && was_full {
                evictions += 1;
            }
            // A converter with no memory whatsoever: fresh cache, one call.
            let want = ToneEnvelopes::default()
                .substitute_magnitudes(id, &params)
                .expect("substituted");
            assert_eq!(
                got.amplitudes_slice(),
                want.amplitudes_slice(),
                "step {i}, tone {id}: a warm cache gave different magnitudes than a cold one"
            );
            assert_eq!(got.omega_0(), want.omega_0());
            assert_eq!(got.harmonic_count(), want.harmonic_count());
            assert_eq!(got.voiced_slice(), want.voiced_slice());
        }
        assert!(
            evictions >= REMEMBERED,
            "only {evictions} evictions — this never exercised a full cache"
        );
    }

    /// A fresh instance and a reused one must agree — no state leaks into the
    /// measurement.
    #[test]
    fn measurement_is_independent_of_converter_history() {
        let params = tone_to_mbe_params(96, 90).expect("params");
        let mut cold = ToneEnvelopes::default();
        let a = cold.substitute_magnitudes(96, &params).expect("cold");

        let mut warm = ToneEnvelopes::default();
        for id in [5u8, 40, 150] {
            let p = tone_to_mbe_params(id, 120).expect("params");
            let _ = warm.substitute_magnitudes(id, &p);
        }
        let b = warm.substitute_magnitudes(96, &params).expect("warm");
        assert_eq!(a.amplitudes_slice(), b.amplitudes_slice());
    }

    /// The silence sentinel has a zero Eq. 209 amplitude and so nothing to
    /// anchor to; it must fall through rather than invent a level.
    #[test]
    fn the_silence_sentinel_is_left_alone() {
        let mut envs = ToneEnvelopes::default();
        let params = tone_to_mbe_params(255, 100).expect("silence params");
        assert!(envs.substitute_magnitudes(255, &params).is_none());
    }
}
