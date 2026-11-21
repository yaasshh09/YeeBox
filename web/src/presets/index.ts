// The presets: a full-size scene for the GPU and a compact twin for the CPU
// fallback (the twin lives in validation/scenes so Tier 4 runs it). Scenes
// are imported as raw text so the exact bytes reach the Rust side.
import doubleSlit from "./double_slit.json?raw";
import lens from "./lens.json?raw";
import waveguide from "./waveguide.json?raw";
import doubleSlitCompact from "../../../validation/scenes/preset_double_slit_compact.json?raw";
import lensCompact from "../../../validation/scenes/preset_lens_compact.json?raw";
import waveguideCompact from "../../../validation/scenes/preset_waveguide_compact.json?raw";

export type Preset = {
  id: string;
  title: string;
  watch: string;
  tryIt: string;
  /** Validation cases that pass and back this preset; empty for none yet. */
  cases: string[];
  full: string;
  compact: string;
};

export const PRESETS: Preset[] = [
  {
    id: "double_slit",
    title: "Double slit",
    watch: "Two slits turn one wave into a fan of bright and dark fringes.",
    tryIt: "Paint over one slit and watch the fringes vanish.",
    cases: [],
    full: doubleSlit,
    compact: doubleSlitCompact,
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
];

export const presetById = (id: string) => PRESETS.find((p) => p.id === id) ?? PRESETS[0];
