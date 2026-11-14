//! Compute pipelines and bind groups. Every pass uses bind group 0 for the
//! per-step `Params` slot (a uniform with a dynamic offset, one slot per
//! step of a batch) and bind group 1 for its storage buffers.

use crate::buffers::Params;
use std::num::NonZeroU64;

/// How a pass uses one storage buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    ReadWrite,
}

/// A compiled pass and the layout its storage bind group must match.
pub struct Pass {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

fn params_size() -> Option<NonZeroU64> {
    NonZeroU64::new(std::mem::size_of::<Params>() as u64)
}

/// The layout of bind group 0, shared by every pass.
pub fn params_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("params"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: params_size(),
            },
            count: None,
        }],
    })
}

/// Bind group 0 over one slot of the slot buffer; the dynamic offset picks
/// the step.
pub fn params_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("params"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: params_size(),
            }),
        }],
    })
}

/// The line every 2D shader declares its workgroup edge on. `pass` swaps
/// it for the requested size, so one source serves 4, 8 and 16.
pub const WG_LINE: &str = "const WG: u32 = 8u;";

/// Compiles `source` with its storage bindings 0, 1, ... used as `access`,
/// and the workgroup edge set to `wg` (2D shaders only; a shader without
/// `WG_LINE` keeps its own size).
pub fn pass(
    device: &wgpu::Device,
    params: &wgpu::BindGroupLayout,
    label: &str,
    source: &str,
    wg: u32,
    access: &[Access],
) -> Pass {
    let source = source.replace(WG_LINE, &format!("const WG: u32 = {wg}u;"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let entries: Vec<_> = access
        .iter()
        .enumerate()
        .map(|(b, a)| wgpu::BindGroupLayoutEntry {
            binding: b as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: *a == Access::Read,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(params), Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Pass { pipeline, layout }
}

/// Bind group 1 for `pass`, buffers in binding order.
pub fn storage_group(
    device: &wgpu::Device,
    pass: &Pass,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(b, buf)| wgpu::BindGroupEntry {
            binding: b as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.layout,
        entries: &entries,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::WG_LINE;

    /// Every shader of the kernel, so the checks below cannot miss one.
    pub(crate) const SHADERS: [(&str, &str); 4] = [
        ("update_h", include_str!("../shaders/update_h.wgsl")),
        ("update_e", include_str!("../shaders/update_e.wgsl")),
        ("sources", include_str!("../shaders/sources.wgsl")),
        ("gather", include_str!("../shaders/gather.wgsl")),
    ];

    fn code(src: &str) -> String {
        src.lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Coefficient formulas live in fdtd-core only. A shader that holds a
    /// float literal or a transcendental function is computing one.
    #[test]
    fn shaders_hold_no_formula() {
        for (name, src) in SHADERS {
            let code = code(src);
            for f in ["exp(", "pow(", "sin(", "cos(", "sqrt(", "log(", "tanh("] {
                assert!(!code.contains(f), "{name} calls {f}");
            }
            for word in code.split(|c: char| !c.is_ascii_alphanumeric() && c != '.') {
                if word.starts_with(|c: char| c.is_ascii_digit()) {
                    let integer = word.trim_end_matches(['u', 'i']);
                    assert!(
                        integer.chars().all(|c| c.is_ascii_digit()),
                        "{name} has a non-integer literal {word}"
                    );
                }
            }
        }
    }

    /// The literal check itself catches what it is meant to catch.
    #[test]
    fn literal_check_catches_floats() {
        for bad in ["0.5", "1e-3", "2.0f", "3f"] {
            let caught = code(bad)
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '.')
                .any(|w| {
                    w.starts_with(|c: char| c.is_ascii_digit())
                        && !w
                            .trim_end_matches(['u', 'i'])
                            .chars()
                            .all(|c| c.is_ascii_digit())
                });
            assert!(caught, "{bad}");
        }
    }

    /// Every 2D shader can have its workgroup edge swapped.
    #[test]
    fn shaders_declare_the_workgroup_line() {
        for (name, src) in &SHADERS[..2] {
            assert!(src.contains(WG_LINE), "{name}");
        }
    }
}
