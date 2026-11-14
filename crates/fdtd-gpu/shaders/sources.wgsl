// Point sources, after the E update and the TF/SF E corrections, exactly
// as fdtd-core/src/tm.rs applies them: in scene order, soft sources add,
// hard sources overwrite. One invocation walks the list, so two sources on
// one node land in the same order as on the CPU. The values themselves are
// Source::value_at(t) for t = (n + 1) S, computed on the CPU in f64.

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
@group(1) @binding(0) var<storage, read_write> ez: array<f32>;
// (node index j * nx + i, 1 for hard)
@group(1) @binding(1) var<storage, read> nodes: array<vec2<u32>>;
// value of source s at batch step b: values[b * n_src + s]
@group(1) @binding(2) var<storage, read> values: array<f32>;

@compute @workgroup_size(1)
fn main() {
    for (var s = 0u; s < p.n_src; s = s + 1u) {
        let node = nodes[s];
        let v = values[p.step * p.n_src + s];
        if (node.y == 1u) {
            ez[node.x] = v;
        } else {
            ez[node.x] = ez[node.x] + v;
        }
    }
}
