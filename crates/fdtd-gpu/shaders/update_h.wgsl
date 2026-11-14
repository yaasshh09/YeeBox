// H update to n + 1/2, CPML included, one invocation per padded index.
// A port of the H loops of fdtd-core/src/tm.rs, same operation order.
// Hx[i, j] at (i, j + 1/2), Hy[i, j] at (i + 1/2, j); nx * ny arrays,
// index j * nx + i. Hx row ny - 1 is padding unless y is periodic; Hy
// column nx - 1 is always padding. Padding is never written.
// Interior profile entries are (0, 0, 1), so psi stays 0 there and every
// line reduces to the plain Yee update.

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
@group(1) @binding(0) var<storage, read> ez: array<f32>;
@group(1) @binding(1) var<storage, read_write> hx: array<f32>;
@group(1) @binding(2) var<storage, read_write> hy: array<f32>;
@group(1) @binding(3) var<storage, read_write> psi_hxy: array<f32>;
@group(1) @binding(4) var<storage, read_write> psi_hyx: array<f32>;
// Packed profiles and material table, see src/buffers.rs.
@group(1) @binding(5) var<storage, read> coef: array<vec4<f32>>;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    // The dispatch overhangs the grid by up to WG - 1 in each direction.
    if (i >= p.nx || j >= p.ny) {
        return;
    }
    let k = j * p.nx + i;

    // Hx: minus the y-difference of Ez, y profile at the half position.
    if (j + 1u < p.ny || p.periodic_y == 1u) {
        let prof = coef[2u * p.nx + p.ny + j];
        var jn = j + 1u;
        if (jn == p.ny) {
            jn = 0u;
        }
        let up = jn * p.nx + i;
        let psi = prof.x * psi_hxy[k] + prof.y * (ez[up] - ez[k]);
        psi_hxy[k] = psi;
        hx[k] = hx[k] - p.s * (ez[up] * prof.z - ez[k] * prof.z + psi);
    }

    // Hy: the x-difference of Ez, x profile at the half position.
    if (i + 1u < p.nx) {
        let prof = coef[p.nx + i];
        let psi = prof.x * psi_hyx[k] + prof.y * (ez[k + 1u] - ez[k]);
        psi_hyx[k] = psi;
        hy[k] = hy[k] + p.s * (ez[k + 1u] * prof.z - ez[k] * prof.z + psi);
    }
}
