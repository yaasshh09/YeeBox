// E update to n + 1, CPML included, interior nodes only. A port of the Ez
// loop of fdtd-core/src/tm.rs, summed in the same order. The outer ring of
// Ez nodes is the PEC wall and is never written (with periodic y every row
// is live and only the x walls remain). PEC material nodes have Ca = Cb = 0
// in the uploaded table.

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
@group(1) @binding(0) var<storage, read_write> ez: array<f32>;
@group(1) @binding(1) var<storage, read> hx: array<f32>;
@group(1) @binding(2) var<storage, read> hy: array<f32>;
@group(1) @binding(3) var<storage, read_write> psi_ezx: array<f32>;
@group(1) @binding(4) var<storage, read_write> psi_ezy: array<f32>;
@group(1) @binding(5) var<storage, read> ids: array<u32>;
// Packed profiles and material table, see src/buffers.rs.
@group(1) @binding(6) var<storage, read> coef: array<vec4<f32>>;

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

    let px = px_prof.x * psi_ezx[k] + px_prof.y * (hy[k] - hy[k - 1u]);
    psi_ezx[k] = px;
    let py = py_prof.x * psi_ezy[k] + py_prof.y * (hx[k] - hx[down]);
    psi_ezy[k] = py;
    let curl = hy[k] * px_prof.z - hy[k - 1u] * px_prof.z - hx[k] * py_prof.z
        + hx[down] * py_prof.z + px - py;
    let m = coef[2u * (p.nx + p.ny) + ids[k]];
    ez[k] = m.x * ez[k] + m.y * curl;
}
