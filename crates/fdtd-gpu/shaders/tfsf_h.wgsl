// TF/SF H corrections, right after the H update: the list from
// fdtd_core::tfsf::Tfsf::h_corrections, field[k] += factor * inc[src],
// with inc the incident Ez of this step (Tfsf::incident_e, uploaded per
// batch). One invocation per group, its two entries in order. Groups
// never share a target, so they run in parallel.

struct Params {
    nx: u32,
    ny: u32,
    periodic_y: u32,
    step: u32,
    s: f32,
    n_src: u32,
    n_taps: u32,
    n_hgroups: u32,
    n_egroups: u32,
    n_inc: u32,
    pad0: u32,
    pad1: u32,
}

// field: 1 Hx, 2 Hy (the numbering of the gather taps)
struct Corr {
    k: u32,
    src: u32,
    field: u32,
    factor: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(1) @binding(0) var<storage, read_write> hx: array<f32>;
@group(1) @binding(1) var<storage, read_write> hy: array<f32>;
@group(1) @binding(2) var<storage, read> corr: array<Corr>;
// incident Ez of batch step b: inc[b * n_inc + src]
@group(1) @binding(3) var<storage, read> inc: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = id.x;
    if (g >= p.n_hgroups) {
        return;
    }
    for (var e = 0u; e < 2u; e = e + 1u) {
        let c = corr[2u * g + e];
        let v = c.factor * inc[p.step * p.n_inc + c.src];
        if (c.field == 1u) {
            hx[c.k] = hx[c.k] + v;
        } else {
            hy[c.k] = hy[c.k] + v;
        }
    }
}
