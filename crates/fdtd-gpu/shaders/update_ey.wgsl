// TEz Ey update to n + 1, CPML included. A port of the Ey loop of
// fdtd-core/src/te.rs: Ey[i, j] at (i + 1/2, j), -dHz/dx, x profile at the
// half position, then the edge's medium row (medium.wgsl, prepended).
// Column nx - 1 is padding and is never written.

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
@group(1) @binding(1) var<storage, read_write> ey: array<f32>;
@group(1) @binding(2) var<storage, read_write> psi_eyx: array<f32>;
// Packed profiles and medium table, see src/buffers.rs.
@group(1) @binding(3) var<storage, read> coef: array<vec4<f32>>;
// Medium row of every Ey edge.
@group(1) @binding(4) var<storage, read> ids: array<u32>;
@group(1) @binding(5) var<storage, read_write> aux: array<f32>;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i + 1u >= p.nx || j >= p.ny) {
        return;
    }
    let k = j * p.nx + i;
    let prof = coef[p.nx + i];
    let psi = prof.x * psi_eyx[k] + prof.y * (hz[k + 1u] - hz[k]);
    psi_eyx[k] = psi;
    let curl = -hz[k + 1u] * prof.z + hz[k] * prof.z - psi;
    ey[k] = medium_update(2u * (p.nx + p.ny) + 3u * ids[k], k, ey[k], curl);
}
