// TF/SF E corrections, right after the E update: the list from
// fdtd_core::tfsf::Tfsf::e_corrections, ez[k] += factor * inc[src], with
// inc the two incident H values of this step (Tfsf::incident_h after the
// auxiliary grid advanced, uploaded per batch) and factor the node's Cb
// with its sign. One invocation per group, entries in order: a box one
// node thick puts both entries of a group on the same node.

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

struct Corr {
    k: u32,
    src: u32,
    field: u32,
    factor: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(1) @binding(0) var<storage, read_write> ez: array<f32>;
@group(1) @binding(1) var<storage, read> corr: array<Corr>;
// incident H of batch step b: inc[2 b + src]
@group(1) @binding(2) var<storage, read> inc: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = id.x;
    if (g >= p.n_egroups) {
        return;
    }
    for (var e = 0u; e < 2u; e = e + 1u) {
        let c = corr[2u * g + e];
        ez[c.k] = ez[c.k] + c.factor * inc[2u * p.step + c.src];
    }
}
