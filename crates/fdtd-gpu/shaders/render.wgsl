// The field display: one full-screen triangle; each pixel finds the node
// under it through the viewport (fdtd-gpu/src/render.rs, Viewport::cell_at,
// y up), colors it from the colormap table, tints the CPML layer, and lays
// the material overlay on top, brighter on material edges.

struct R {
    nx: u32,
    ny: u32,
    pml: u32,
    periodic_y: u32,
    cx: f32,
    cy: f32,
    px_per_cell: f32,
    gain: f32,
    width: f32,
    height: f32,
    mode: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var<storage, read> ez: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
// 257 diverging entries, then 257 sequential, RGBA8 with red in the low byte.
@group(0) @binding(3) var<storage, read> luts: array<u32>;
// Overlay color and opacity per material id, the wall entry last.
@group(0) @binding(4) var<storage, read> mats: array<vec4<f32>>;
// Running mean of Ez^2 (intensity view).
@group(0) @binding(5) var<storage, read> mean: array<f32>;

const PML_TINT: vec3<f32> = vec3<f32>(0.20, 0.22, 0.30);

@vertex
fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let x = f32((v << 1u) & 2u);
    let y = f32(v & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

fn unpack(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let x = r.cx + (p.x - 0.5 * r.width) / r.px_per_cell;
    let y = r.cy - (p.y - 0.5 * r.height) / r.px_per_cell;
    let fi = round(x);
    let fj = round(y);
    if (fi < 0.0 || fj < 0.0 || fi > f32(r.nx - 1u) || fj > f32(r.ny - 1u)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let i = u32(fi);
    let j = u32(fj);
    let k = j * r.nx + i;
    var c: vec3<f32>;
    if (r.mode == 1u) {
        // A sinusoid of amplitude 1 / gain has mean square 1 / (2 gain^2).
        let u = clamp(2.0 * mean[k] * r.gain * r.gain, 0.0, 1.0);
        c = unpack(luts[257u + u32(round(u * 256.0))]);
    } else {
        let v = clamp(ez[k] * r.gain, -1.0, 1.0);
        c = unpack(luts[u32(round((v + 1.0) * 128.0))]);
    }
    let d = r.pml;
    let in_y = r.periodic_y == 0u && (j < d || j > r.ny - 1u - d);
    if (d > 0u && (i < d || i > r.nx - 1u - d || in_y)) {
        c = mix(c, PML_TINT, 0.35);
    }
    let id = ids[k];
    var m = mats[id];
    let edge = (i > 0u && ids[k - 1u] != id) || (i + 1u < r.nx && ids[k + 1u] != id)
        || (j > 0u && ids[k - r.nx] != id) || (j + 1u < r.ny && ids[k + r.nx] != id);
    if (edge && m.a > 0.0) {
        m = vec4<f32>(m.rgb, max(m.a, 0.8));
    }
    c = mix(c, m.rgb, m.a);
    return vec4<f32>(c, 1.0);
}
