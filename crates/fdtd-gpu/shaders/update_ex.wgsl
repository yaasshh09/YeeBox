// TEz Ex update to n + 1, CPML included. A port of the Ex loop of
// fdtd-core/src/te.rs: Ex[i, j] at (i, j + 1/2), dHz/dy, y profile at the
// half position, then the edge's medium row (medium.wgsl, prepended). Row
// ny - 1 is padding unless y is periodic, and is never written then.

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

const WG: u32 = 8u;

@group(0) @binding(0) var<uniform> p: Params;
@group(1) @binding(0) var<storage, read> hz: array<f32>;
@group(1) @binding(1) var<storage, read_write> ex: array<f32>;
@group(1) @binding(2) var<storage, read_write> psi_exy: array<f32>;
// Packed profiles and medium table, see src/buffers.rs.
@group(1) @binding(3) var<storage, read> coef: array<vec4<f32>>;
// Medium row of every Ex edge.
@group(1) @binding(4) var<storage, read> ids: array<u32>;
@group(1) @binding(5) var<storage, read_write> aux: array<f32>;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= p.nx || j >= p.ny) {
        return;
    }
    if (j + 1u >= p.ny && p.periodic_y == 0u) {
        return;
    }
    var jn = j + 1u;
    if (jn == p.ny) {
        jn = 0u;
    }
    let k = j * p.nx + i;
    let up = jn * p.nx + i;
    let prof = coef[2u * p.nx + p.ny + j];
    let psi = prof.x * psi_exy[k] + prof.y * (hz[up] - hz[k]);
    psi_exy[k] = psi;
    let curl = hz[up] * prof.z - hz[k] * prof.z + psi;
    ex[k] = medium_update(2u * (p.nx + p.ny) + 3u * ids[k], k, ex[k], curl);
}
