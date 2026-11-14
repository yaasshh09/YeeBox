//! A small line plot written straight to SVG, so `fdtd report` needs no
//! plotting stack and its output diffs cleanly. Linear axes, one polyline
//! with markers per series, dashed horizontal reference lines, a legend.

use std::fmt::Write;

pub struct Series<'a> {
    pub label: &'a str,
    pub color: &'a str,
    pub points: &'a [(f64, f64)],
}

pub struct Plot<'a> {
    pub title: &'a str,
    pub x_label: &'a str,
    pub y_label: &'a str,
    /// Axis range and tick step: (min, max, step).
    pub x: (f64, f64, f64),
    pub y: (f64, f64, f64),
    pub series: &'a [Series<'a>],
    /// Dashed horizontal lines: (y, label).
    pub hlines: &'a [(f64, &'a str)],
}

const W: f64 = 640.0;
const H: f64 = 420.0;
const LEFT: f64 = 70.0;
const RIGHT: f64 = 20.0;
const TOP: f64 = 40.0;
const BOTTOM: f64 = 55.0;

/// Escapes the characters that are special in SVG text.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Tick values from min to max inclusive, without float drift.
fn ticks((min, max, step): (f64, f64, f64)) -> Vec<f64> {
    let n = ((max - min) / step).round() as i64;
    (0..=n).map(|k| min + k as f64 * step).collect()
}

/// A tick label: `t` rounded to the decimals the step implies, trailing
/// zeros dropped, so 3 x 0.2 prints as 0.6, not 0.6000000000000001.
fn label(t: f64, step: f64) -> String {
    let decimals = (-step.abs().log10().floor()).max(0.0) as usize;
    let s = format!("{t:.decimals$}");
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    if s == "-0" { "0".to_string() } else { s }
}

impl Plot<'_> {
    fn px(&self, x: f64) -> f64 {
        LEFT + (x - self.x.0) / (self.x.1 - self.x.0) * (W - LEFT - RIGHT)
    }

    fn py(&self, y: f64) -> f64 {
        TOP + (self.y.1 - y) / (self.y.1 - self.y.0) * (H - TOP - BOTTOM)
    }

    /// The whole document. Writing to a `String` cannot fail.
    pub fn render(&self) -> String {
        let mut s = String::new();
        let (x0, x1, y0, y1) = (LEFT, W - RIGHT, TOP, H - BOTTOM);
        let _ = writeln!(
            s,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">"#
        );
        let _ = writeln!(s, r#"<rect width="{W}" height="{H}" fill="white"/>"#);
        let _ = writeln!(
            s,
            r#"<text x="{}" y="22" text-anchor="middle" font-size="14">{}</text>"#,
            W / 2.0,
            esc(self.title)
        );
        for t in ticks(self.x) {
            let x = self.px(t);
            let _ = writeln!(
                s,
                r##"<line x1="{x:.1}" y1="{y0}" x2="{x:.1}" y2="{y1}" stroke="#e4e4e4"/><text x="{x:.1}" y="{}" text-anchor="middle">{}</text>"##,
                y1 + 16.0,
                label(t, self.x.2)
            );
        }
        for t in ticks(self.y) {
            let y = self.py(t);
            let _ = writeln!(
                s,
                r##"<line x1="{x0}" y1="{y:.1}" x2="{x1}" y2="{y:.1}" stroke="#e4e4e4"/><text x="{}" y="{:.1}" text-anchor="end">{}</text>"##,
                x0 - 6.0,
                y + 4.0,
                label(t, self.y.2)
            );
        }
        let _ = writeln!(
            s,
            r#"<rect x="{x0}" y="{y0}" width="{}" height="{}" fill="none" stroke="black"/>"#,
            x1 - x0,
            y1 - y0
        );
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{}" text-anchor="middle">{}</text>"#,
            (x0 + x1) / 2.0,
            H - 12.0,
            esc(self.x_label)
        );
        let _ = writeln!(
            s,
            r#"<text transform="translate(18 {}) rotate(-90)" text-anchor="middle">{}</text>"#,
            (y0 + y1) / 2.0,
            esc(self.y_label)
        );
        for &(y, label) in self.hlines {
            let py = self.py(y);
            let _ = writeln!(
                s,
                r##"<line x1="{x0}" y1="{py:.1}" x2="{x1}" y2="{py:.1}" stroke="#c0392b" stroke-dasharray="6 4"/><text x="{}" y="{:.1}" text-anchor="end" fill="#c0392b">{}</text>"##,
                x1 - 6.0,
                py - 5.0,
                esc(label)
            );
        }
        for (k, ser) in self.series.iter().enumerate() {
            let pts: Vec<String> = ser
                .points
                .iter()
                .map(|&(x, y)| format!("{:.1},{:.1}", self.px(x), self.py(y)))
                .collect();
            let _ = writeln!(
                s,
                r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
                pts.join(" "),
                ser.color
            );
            for &(x, y) in ser.points {
                let _ = writeln!(
                    s,
                    r#"<circle cx="{:.1}" cy="{:.1}" r="3" fill="{}"/>"#,
                    self.px(x),
                    self.py(y),
                    ser.color
                );
            }
            let ly = y0 + 18.0 + 18.0 * k as f64;
            let _ = writeln!(
                s,
                r#"<line x1="{}" y1="{ly}" x2="{}" y2="{ly}" stroke="{}" stroke-width="2"/><text x="{}" y="{}">{}</text>"#,
                x0 + 12.0,
                x0 + 36.0,
                ser.color,
                x0 + 42.0,
                ly + 4.0,
                esc(ser.label)
            );
        }
        s.push_str("</svg>\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_axes_and_writes_series() {
        let pts = [(0.0, -100.0), (80.0, -60.0)];
        let series = [Series {
            label: "a < b",
            color: "blue",
            points: &pts,
        }];
        let plot = Plot {
            title: "t",
            x_label: "x",
            y_label: "y",
            x: (0.0, 80.0, 10.0),
            y: (-140.0, 0.0, 20.0),
            series: &series,
            hlines: &[(-60.0, "gate")],
        };
        // Corners of the data area.
        assert_eq!(plot.px(0.0), LEFT);
        assert_eq!(plot.px(80.0), W - RIGHT);
        assert_eq!(plot.py(0.0), TOP);
        assert_eq!(plot.py(-140.0), H - BOTTOM);
        let svg = plot.render();
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>\n"));
        assert_eq!(svg.matches("<polyline").count(), 1);
        assert_eq!(svg.matches("<circle").count(), 2);
        assert!(svg.contains("a &lt; b") && svg.contains("gate"));
        assert_eq!(ticks((0.0, 80.0, 10.0)).len(), 9);
        assert_eq!(label(3.0 * 0.2, 0.2), "0.6");
        assert_eq!(label(0.005 * 2.0, 0.005), "0.01");
        assert_eq!(label(-140.0, 20.0), "-140");
        assert_eq!(label(-0.0, 0.2), "0");
    }
}
