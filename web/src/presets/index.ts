// The presets: a full-size scene for the GPU and a compact twin for the CPU
// fallback (the twin lives in validation/scenes so Tier 4 runs it). Scenes
// are imported as raw text so the exact bytes reach the Rust side.
import doubleSlit from "./double_slit.json?raw";
import lens from "./lens.json?raw";
import waveguide from "./waveguide.json?raw";
import doubleSlitCompact from "../../../validation/scenes/preset_double_slit_compact.json?raw";
import lensCompact from "../../../validation/scenes/preset_lens_compact.json?raw";
import waveguideCompact from "../../../validation/scenes/preset_waveguide_compact.json?raw";
import plasmon from "./plasmon.json?raw";
import mirror from "./mirror.json?raw";
import absorber from "./absorber.json?raw";
import plasmonCompact from "../../../validation/scenes/preset_plasmon_compact.json?raw";
import mirrorCompact from "../../../validation/scenes/preset_mirror_compact.json?raw";
import absorberCompact from "../../../validation/scenes/preset_absorber_compact.json?raw";
import scattering from "./scattering.json?raw";
import array from "./array.json?raw";
import scatteringCompact from "../../../validation/scenes/preset_scattering_compact.json?raw";
import arrayCompact from "../../../validation/scenes/preset_array_compact.json?raw";
import cylinderPlot from "../../../validation/report/drude_cylinder.png";
import scatteringPlot from "../../../validation/report/cylinder_scattering.svg";

export type Preset = {
  id: string;
  title: string;
  watch: string;
  tryIt: string;
  /** Validation cases that pass and back this preset; empty for none yet. */
  cases: string[];
  /** A plot of the case, opened by the badge instead of the Verify panel. */
  plot?: string;
  /** Shows a steering slider on the polar plot (CW arrays). */
  steer?: boolean;
  full: string;
  compact: string;
};

export const PRESETS: Preset[] = [
  {
    id: "double_slit",
    title: "Double slit",
    watch: "Two slits turn one wave into a fan of bright and dark fringes.",
    tryIt: "Paint over one slit and watch the fringes vanish.",
    cases: ["A22"],
    full: doubleSlit,
    compact: doubleSlitCompact,
  },
  {
    id: "scattering",
    title: "Scattering cylinder",
    watch: "A metal cylinder in a plane wave casts a shadow and throws a bright forward lobe; the polar plot sits on the analytic series.",
    tryIt: "Paint over the cylinder to change its shape and watch the pattern leave the dashed curve.",
    cases: ["A19"],
    plot: scatteringPlot,
    full: scattering,
    compact: scatteringCompact,
  },
  {
    id: "array",
    title: "Phased array",
    watch: "Eight sources half a wavelength apart send one beam where their phases point it.",
    tryIt: "Drag the steer slider under the polar plot and the beam swings with the dashed array factor.",
    cases: ["A23"],
    steer: true,
    full: array,
    compact: arrayCompact,
  },
  {
    id: "waveguide",
    title: "Waveguide and cutoff",
    watch: "A guided mode runs between two metal plates.",
    tryIt: "Raise the wavelength past 120 cells and the guide stops carrying it.",
    cases: ["A11", "A12"],
    full: waveguide,
    compact: waveguideCompact,
  },
  {
    id: "lens",
    title: "Lens",
    watch: "A plane wave focuses to a spot behind a glass lens.",
    tryIt: "Paint high index glass over the lens and the focus moves closer.",
    cases: ["A13"],
    full: lens,
    compact: lensCompact,
  },
  {
    id: "plasmon",
    title: "Plasmon on a silver wire",
    watch: "Light at 360 nm drives the electrons of a 24 nm silver wire, which bends the wave far more than its size suggests.",
    tryIt: "Switch to TM and the plasmon is gone: it needs the electric field in the plane.",
    cases: ["A21"],
    plot: cylinderPlot,
    full: plasmon,
    compact: plasmonCompact,
  },
  {
    id: "mirror",
    title: "Silver mirror",
    watch: "A source in front of a silver wall: the reflection stands still as a pattern of bands.",
    tryIt: "Paint silver to bend the mirror and watch the bands follow.",
    cases: ["A16"],
    full: mirror,
    compact: mirrorCompact,
  },
  {
    id: "absorber",
    title: "Resonant absorber",
    watch: "Rods tuned to the light soak it up and leave a shadow behind them.",
    tryIt: "Change the wavelength and the rods go transparent.",
    cases: ["A17"],
    full: absorber,
    compact: absorberCompact,
  },
];

export const presetById = (id: string) => PRESETS.find((p) => p.id === id) ?? PRESETS[0];
