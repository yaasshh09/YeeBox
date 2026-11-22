// TEz Hz update to n + 1/2, CPML included, interior nodes only. A port of
// the Hz loop of fdtd-core/src/te.rs, summed in the same order. The ring of
// Hz nodes is the magnetic wall and is never written (with periodic y every
// row is live and only the x walls remain). Hz has mu = 1: coefficient S.

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
@group(1) @binding(0) var<storage, read_write> hz: array<f32>;
@group(1) @binding(1) var<storage, read> ex: array<f32>;
@group(1) @binding(2) var<storage, read> ey: array<f32>;
@group(1) @binding(3) var<storage, read_write> psi_hzx: array<f32>;
@group(1) @binding(4) var<storage, read_write> psi_hzy: array<f32>;
// Packed profiles and medium table, see src/buffers.rs.
@group(1) @binding(5) var<storage, read> coef: array<vec4<f32>>;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i == 0u || i + 1u >= p.nx || j >= p.ny) {
        return;
    }
    if (p.periodic_y == 0u && (j == 0u || j + 1u >= p.ny)) {
        return;
    }
    var jp = p.ny - 1u;
    if (j > 0u) {
        jp = j - 1u;
    }
    let k = j * p.nx + i;
    let down = jp * p.nx + i;
    let px_prof = coef[i];
    let py_prof = coef[2u * p.nx + j];

    let px = px_prof.x * psi_hzx[k] + px_prof.y * (ey[k] - ey[k - 1u]);
    psi_hzx[k] = px;
    let py = py_prof.x * psi_hzy[k] + py_prof.y * (ex[k] - ex[down]);
    psi_hzy[k] = py;
    let curl = -ey[k] * px_prof.z + ey[k - 1u] * px_prof.z + ex[k] * py_prof.z
        - ex[down] * py_prof.z - px + py;
    hz[k] = hz[k] + p.s * curl;
}
