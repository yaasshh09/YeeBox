//! The scene as the user edits it, shared by the GPU and CPU bindings: it
//! turns tool actions (in palette indices and cell positions) into core
//! scene edits and says what a kernel must update. Every rule lives in
//! `fdtd-core`; this only routes.

use fdtd_core::scene::{
    MAX_WAVELENGTH_CELLS, MIN_WAVELENGTH_CELLS, Scene, Shape, SlitPair, Source, SourcePatch,
    StrokeOutcome, slit_pair,
};
use fdtd_core::source::{SourceKind, Waveform};
use fdtd_core::{FdtdError, Result};

/// A scene with the paint palette merged into its table.
#[derive(Debug, Clone)]
pub struct Session {
    pub scene: Scene,
    /// Table index of each palette entry, in `material::PALETTE` order.
    palette: Vec<usize>,
}

impl Session {
    /// Parses and validates a scene and merges the palette into its table.
    pub fn load(json: &str) -> Result<Session> {
        let mut scene = Scene::from_json(json)?;
        let palette = scene.with_palette();
        scene.validate()?;
        Ok(Session { scene, palette })
    }

    fn material(&self, palette_index: usize) -> Result<usize> {
        self.palette
            .get(palette_index)
            .copied()
            .ok_or(FdtdError::InvalidParameter {
                name: "palette index",
                value: palette_index as f64,
            })
    }

    /// One brush stroke with palette entry `palette_index` (0 erases).
    pub fn paint(
        &mut self,
        from: (f64, f64),
        to: (f64, f64),
        radius: f64,
        palette_index: usize,
    ) -> Result<StrokeOutcome> {
        let m = self.material(palette_index)?;
        self.scene.paint_stroke(from, to, radius, m)
    }

    /// The node bounds of a shape, clipped to the grid.
    fn node_bounds(&self, b: [f64; 4]) -> [usize; 4] {
        let (nx, ny) = (self.scene.grid.nx as f64, self.scene.grid.ny as f64);
        let lo = |v: f64, n: f64| v.floor().clamp(0.0, n - 1.0) as usize;
        let hi = |v: f64, n: f64| v.ceil().clamp(0.0, n - 1.0) as usize;
        [lo(b[0], nx), lo(b[1], ny), hi(b[2], nx), hi(b[3], ny)]
    }

    /// Adds shapes, keeping the scene valid: on any error they are taken
    /// out again. Returns the node bounds to update.
    fn add_shapes(&mut self, shapes: Vec<Shape>) -> Result<[usize; 4]> {
        let before = self.scene.geometry.len();
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for s in &shapes {
            let sb = s.bounds();
            b = [
                b[0].min(sb[0]),
                b[1].min(sb[1]),
                b[2].max(sb[2]),
                b[3].max(sb[3]),
            ];
        }
        self.scene.geometry.extend(shapes);
        if let Err(e) = self.scene.validate() {
            self.scene.geometry.truncate(before);
            return Err(e);
        }
        Ok(self.node_bounds(b))
    }

    /// Adds a rect, circle or polygon whose `material` is a palette index.
    pub fn add_shape(&mut self, mut shape: Shape) -> Result<[usize; 4]> {
        let m = self.material(shape.material())?;
        match &mut shape {
            Shape::Rect { material, .. }
            | Shape::Circle { material, .. }
            | Shape::Polygon { material, .. } => *material = m,
        }
        self.add_shapes(vec![shape])
    }

    /// Adds a slit pair whose `material` is a palette index.
    pub fn add_slit_pair(&mut self, spec: &SlitPair) -> Result<[usize; 4]> {
        let spec = SlitPair {
            material: self.material(spec.material)?,
            ..*spec
        };
        self.add_shapes(slit_pair(&spec)?)
    }

    /// Removes every shape and all paint; sources and the plane wave stay.
    /// Returns the whole grid as the bounds to update.
    pub fn clear(&mut self) -> [usize; 4] {
        self.scene.geometry.clear();
        self.scene.paint = None;
        [0, 0, self.scene.grid.nx - 1, self.scene.grid.ny - 1]
    }

    /// Adds a soft CW point source at the node nearest (x, y), turning on
    /// from normalized time `t_now` over 3 periods.
    pub fn add_source(&mut self, x: f64, y: f64, wavelength_cells: f64, t_now: f64) -> Result<()> {
        if !(MIN_WAVELENGTH_CELLS..=MAX_WAVELENGTH_CELLS).contains(&wavelength_cells) {
            return Err(FdtdError::InvalidParameter {
                name: "wavelength_cells",
                value: wavelength_cells,
            });
        }
        if !(x.is_finite() && y.is_finite() && x >= 0.0 && y >= 0.0) {
            return Err(FdtdError::OutOfDomain {
                what: "source",
                i: 0,
                j: 0,
            });
        }
        let src = Source {
            i: x.round() as usize,
            j: y.round() as usize,
            waveform: Waveform::RampedCw {
                f0: 1.0 / wavelength_cells,
                ramp_periods: 3.0,
                phase: 0.0,
                start: t_now.max(0.0),
            },
            amplitude: 1.0,
            kind: SourceKind::Soft,
        };
        self.scene.check_source(&src)?;
        self.scene.sources.push(src);
        Ok(())
    }

    pub fn remove_source(&mut self, index: usize) -> Result<()> {
        if index >= self.scene.sources.len() {
            return Err(FdtdError::InvalidParameter {
                name: "source index",
                value: index as f64,
            });
        }
        self.scene.sources.remove(index);
        Ok(())
    }

    pub fn edit_source(&mut self, index: usize, patch: &SourcePatch, t_now: f64) -> Result<()> {
        self.scene.patch_source(index, patch, t_now)
    }

    /// The source nearest (x, y) within `within` cells, for picking one up.
    pub fn source_near(&self, x: f64, y: f64, within: f64) -> Option<usize> {
        self.scene
            .sources
            .iter()
            .enumerate()
            .map(|(k, s)| (k, (s.i as f64 - x).hypot(s.j as f64 - y)))
            .filter(|&(_, d)| d <= within)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(k, _)| k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fdtd_core::material::PALETTE;

    /// 60 x 50 vacuum with a 5-cell layer and one CW source at (20, 25).
    const SCENE: &str = r#"{"version": 1,
        "grid": {"nx": 60, "ny": 50, "dx_phys_m": 0.001, "courant": 0.5},
        "materials": [{"kind": "dielectric", "eps_inf": 1, "sigma": 0}],
        "sources": [{"i": 20, "j": 25, "amplitude": 1,
            "waveform": {"type": "ramped_cw", "f0": 0.05, "ramp_periods": 3, "phase": 0}}],
        "pml": {"thickness": 5, "m": 3, "sigma_scale": 1.25, "kappa_max": 2, "alpha_max": 0.15}}"#;

    #[test]
    fn load_merges_the_palette() {
        let s = Session::load(SCENE).unwrap();
        assert_eq!(s.scene.materials.len(), PALETTE.len());
        for (k, (_, m)) in PALETTE.iter().enumerate() {
            assert_eq!(s.scene.materials[s.palette[k]], *m);
        }
        assert!(Session::load("{").is_err());
    }

    #[test]
    fn paint_uses_palette_indices() {
        let mut s = Session::load(SCENE).unwrap();
        let out = s.paint((30.0, 10.0), (30.0, 40.0), 0.0, 3).unwrap();
        assert_eq!(out.painted, 31);
        let pec = s.palette[3];
        assert_eq!(s.scene.material_at(30, 20), pec);
        assert!(s.paint((1.0, 1.0), (2.0, 2.0), 1.0, 99).is_err());
    }

    #[test]
    fn shapes_land_or_are_taken_back() {
        let mut s = Session::load(SCENE).unwrap();
        let rect = Shape::Rect {
            x0: 30.0,
            y0: 10.0,
            x1: 34.5,
            y1: 20.0,
            material: 1,
        };
        assert_eq!(s.add_shape(rect).unwrap(), [30, 10, 35, 20]);
        assert_eq!(s.scene.material_at(32, 15), s.palette[1]);
        // Over the source as a conductor: refused and removed again.
        let over = Shape::Circle {
            cx: 20.0,
            cy: 25.0,
            r: 3.0,
            material: 3,
        };
        assert_eq!(
            s.add_shape(over),
            Err(FdtdError::SourceOnPec { i: 20, j: 25 })
        );
        assert_eq!(s.scene.geometry.len(), 1);
        let slits = SlitPair {
            x: 40,
            thickness: 2,
            y0: 5,
            y1: 44,
            center: 25.0,
            separation: 10.0,
            width: 3.0,
            material: 3,
        };
        assert_eq!(s.add_slit_pair(&slits).unwrap(), [40, 5, 41, 44]);
        assert_eq!(s.scene.geometry.len(), 4);
        assert_eq!(s.clear(), [0, 0, 59, 49]);
        assert!(s.scene.geometry.is_empty() && s.scene.paint.is_none());
        assert_eq!(s.scene.sources.len(), 1);
    }

    #[test]
    fn sources_add_late_and_can_be_picked() {
        let mut s = Session::load(SCENE).unwrap();
        s.add_source(40.4, 30.6, 15.0, 250.0).unwrap();
        let added = s.scene.sources[1];
        assert_eq!((added.i, added.j), (40, 31));
        assert!(
            matches!(added.waveform, Waveform::RampedCw { start, f0, .. }
            if start == 250.0 && f0 == 1.0 / 15.0)
        );
        assert!(s.add_source(2.0, 25.0, 15.0, 0.0).is_err(), "in the layer");
        assert!(
            s.add_source(30.0, 25.0, 3.0, 0.0).is_err(),
            "below the grid cutoff"
        );
        assert_eq!(s.source_near(39.0, 31.5, 2.0), Some(1));
        assert_eq!(s.source_near(30.0, 30.0, 2.0), None);
        s.remove_source(0).unwrap();
        assert_eq!(s.scene.sources.len(), 1);
        assert!(s.remove_source(5).is_err());
        s.scene.validate().unwrap();
    }
}
