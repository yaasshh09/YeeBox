// The running mean of Ez^2 the intensity view shows, last pass of a step
// and only while that view is on: mean += alpha * (ez^2 - mean). A display
// quantity; no kernel reads it. alpha is 1 / tau, tau in steps.

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

@group(0) @binding(0) var<uniform> p: Params;
@group(1) @binding(0) var<storage, read> ez: array<f32>;
@group(1) @binding(1) var<storage, read_write> mean: array<f32>;
@group(1) @binding(2) var<storage, read> alpha: array<f32>;

const WG: u32 = 8u;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.nx || id.y >= p.ny) {
        return;
    }
    let k = id.y * p.nx + id.x;
    let e = ez[k];
    mean[k] = mean[k] + alpha[0] * (e * e - mean[k]);
}
