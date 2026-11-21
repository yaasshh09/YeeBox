//! The two colormaps the field display uses, as 257-entry RGBA8 tables in
//! sRGB, shared by the GPU render pass and the CPU fallback so both draw
//! the same pixels. A display transform only: no kernel reads them.
//!
//! Built in OKLab from rules rather than copied from a published table:
//!
//! - `diverging` (signed fields): blue below zero, orange above, through a
//!   dark neutral at zero that sits on the black stage. Lightness rises
//!   with |v| and is the same for v and -v. Blue and orange differ along
//!   the OKLab b axis, the one red-green color deficiencies keep, so the
//!   sign stays readable for them.
//! - `sequential` (intensity): dark neutral, through orange, to near white,
//!   lightness rising throughout.
//!
//! Entry 128 is v = 0; a value v in [-1, 1] maps to entry
//! `round((v + 1) * 128)`, and an intensity u in [0, 1] to `round(u * 256)`.

/// Entries per table.
pub const LEN: usize = 257;

/// OKLab lightness of the zero entry: a little above the black stage.
const L_ZERO: f64 = 0.12;
/// OKLab lightness at full scale.
const L_FULL: f64 = 0.90;
/// Hues in OKLCh, radians: a sky blue and an orange.
const HUE_NEG: f64 = 4.45;
const HUE_POS: f64 = 0.95;
/// Chroma at full scale before the gamut fit.
const C_FULL: f64 = 0.16;

/// Linear sRGB of an OKLab color (Ottosson's matrices).
fn oklab_to_linear(l: f64, a: f64, b: f64) -> [f64; 3] {
    let l_ = l + 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let m_ = l - 0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let s_ = l - 0.089_484_177_5 * a - 1.291_485_548_0 * b;
    let (l3, m3, s3) = (l_.powi(3), m_.powi(3), s_.powi(3));
    [
        4.076_741_662_1 * l3 - 3.307_711_591_3 * m3 + 0.230_969_929_2 * s3,
        -1.268_438_004_6 * l3 + 2.609_757_401_1 * m3 - 0.341_319_396_5 * s3,
        -0.004_196_086_3 * l3 - 0.703_418_614_7 * m3 + 1.707_614_701_0 * s3,
    ]
}

fn encode(c: f64) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// The sRGB color of OKLCh (l, c, h), with the chroma cut down by
/// bisection until it fits the gamut. Lightness and hue are kept.
fn lch(l: f64, c: f64, h: f64) -> [u8; 4] {
    let inside = |c: f64| {
        oklab_to_linear(l, c * h.cos(), c * h.sin())
            .iter()
            .all(|v| (-1e-9..=1.0 + 1e-9).contains(v))
    };
    let (mut lo, mut hi) = (0.0, c);
    if !inside(hi) {
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if inside(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi = lo;
    }
    let [r, g, b] = oklab_to_linear(l, hi * h.cos(), hi * h.sin());
    [encode(r), encode(g), encode(b), 255]
}

/// Lightness at magnitude `t` in [0, 1]: rises from `L_ZERO` to `L_FULL`.
fn lightness(t: f64) -> f64 {
    L_ZERO + (L_FULL - L_ZERO) * t.powf(0.8)
}

/// The signed-field table, entry 128 at zero.
pub fn diverging() -> Vec<[u8; 4]> {
    (0..LEN)
        .map(|k| {
            let v = k as f64 / 128.0 - 1.0;
            let t = v.abs();
            let hue = if v < 0.0 { HUE_NEG } else { HUE_POS };
            lch(lightness(t), C_FULL * t, hue)
        })
        .collect()
}

/// The intensity table, entry 0 at zero.
pub fn sequential() -> Vec<[u8; 4]> {
    (0..LEN)
        .map(|k| {
            let t = k as f64 / (LEN - 1) as f64;
            // Orange chroma peaks halfway and fades toward white; the hue
            // drifts toward yellow as it brightens.
            let c = C_FULL * (std::f64::consts::PI * t).sin();
            lch(lightness(t), c, HUE_POS + 0.5 * t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative luminance of an sRGB8 color.
    fn luminance(c: [u8; 4]) -> f64 {
        let lin = |u: u8| {
            let s = u as f64 / 255.0;
            if s <= 0.040_45 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])
    }

    #[test]
    fn zero_is_a_dark_neutral() {
        let d = diverging();
        assert_eq!(d.len(), LEN);
        let z = d[128];
        assert!(z[0] == z[1] && z[1] == z[2], "{z:?}");
        assert!(z[0] < 40, "{z:?}");
        assert_eq!(sequential()[0], z);
    }

    #[test]
    fn lightness_rises_with_magnitude_on_both_sides() {
        let d = diverging();
        for k in 0..128 {
            let (up, down) = (luminance(d[128 + k + 1]), luminance(d[128 - k - 1]));
            assert!(up >= luminance(d[128 + k]), "positive side at {k}");
            assert!(down >= luminance(d[128 - k]), "negative side at {k}");
        }
        let s = sequential();
        for k in 0..LEN - 1 {
            assert!(luminance(s[k + 1]) >= luminance(s[k]), "sequential at {k}");
        }
        assert!(luminance(s[LEN - 1]) > 0.6);
    }

    #[test]
    fn signs_differ_on_the_blue_yellow_axis() {
        // The same OKLab lightness on both sides, so only hue tells the
        // sign; and that hue difference is in blue against yellow, which
        // protanopes and deuteranopes keep.
        let d = diverging();
        for k in [16usize, 64, 128] {
            let (neg, pos) = (d[128 - k], d[128 + k]);
            assert!(neg[2] > neg[0], "negative is blue at {k}: {neg:?}");
            assert!(pos[0] > pos[2], "positive is orange at {k}: {pos:?}");
            let (ln, lp) = (luminance(neg), luminance(pos));
            assert!((ln - lp).abs() / lp < 0.15, "luminance {ln} vs {lp} at {k}");
        }
    }
}
