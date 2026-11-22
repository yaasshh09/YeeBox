// The E update of one location through its medium table row (src/buffers.rs),
// prepended to every E shader. A port of fdtd-core Aux::update, same order:
// P^(n+1) first from E^n, then E^(n+1), then J^(n+1) from E^n and E^(n+1).
// A plain row (flags 0) runs the plain line, the Phase 5 update bit for bit.
// The including shader declares `p`, `coef` and `aux` (J at k, P at n + k,
// P^(n-1) at 2 n + k, n = nx ny).

fn medium_update(row: u32, k: u32, e: f32, curl: f32) -> f32 {
    let m = coef[row];
    let flags = u32(m.z);
    if (flags == 0u) {
        return m.x * e + m.y * curl;
    }
    let n = p.nx * p.ny;
    let d = coef[row + 1u];
    let l = coef[row + 2u];
    var j: f32;
    if ((flags & 1u) != 0u) {
        j = d.z * aux[k];
    }
    var dp: f32;
    if ((flags & 2u) != 0u) {
        let pn = l.x * aux[n + k] + l.y * aux[2u * n + k] + l.z * e;
        dp = l.w * (pn - aux[n + k]);
        aux[2u * n + k] = aux[n + k];
        aux[n + k] = pn;
    }
    let en = m.x * e + m.y * (curl - j) - dp;
    if ((flags & 1u) != 0u) {
        aux[k] = d.x * aux[k] + d.y * (en + e);
    }
    return en;
}
