// Monitor taps, last pass of a step. Copies the field samples every monitor
// reads (fdtd_core::monitor::Monitor::taps) into out[step * n_taps + t].
// The CPU reads them back once per batch and accumulates probes, DFTs and
// flux in f64 with the same code the CPU kernel runs. No arithmetic here.

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
@group(1) @binding(1) var<storage, read> hx: array<f32>;
@group(1) @binding(2) var<storage, read> hy: array<f32>;
// (component: 0 Ez, 1 Hx, 2 Hy; index j * nx + i)
@group(1) @binding(3) var<storage, read> taps: array<vec2<u32>>;
@group(1) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if (t >= p.n_taps) {
        return;
    }
    let tap = taps[t];
    var v = ez[tap.y];
    if (tap.x == 1u) {
        v = hx[tap.y];
    } else if (tap.x == 2u) {
        v = hy[tap.y];
    }
    out[p.step * p.n_taps + t] = v;
}
