pub mod segmentation;
pub mod speaker_identification;

// Backend selection.
//
// On macOS we default to the WGPU backend, which maps to Metal on Apple
// Silicon / AMD GPUs. On every other platform (and when the `wgpu` feature is
// explicitly disabled) we fall back to the portable NdArray CPU backend.
//
// The neural-network model definitions are generic over `B: Backend`, so the
// bundled `.bpk` weights load unchanged regardless of which backend is active.

#[cfg(all(target_os = "macos", feature = "wgpu"))]
mod backend {
    use burn_wgpu::{Wgpu, WgpuDevice};

    /// GPU (Metal via WGPU) backend used for inference with Burn models.
    pub type BurnBackend = Wgpu;
    /// GPU (Metal via WGPU) device used for inference with Burn models.
    pub type BurnDevice = WgpuDevice;
}

#[cfg(not(all(target_os = "macos", feature = "wgpu")))]
mod backend {
    use crate::burn_facade::backend::ndarray::{NdArray, NdArrayDevice};

    /// CPU (NdArray) backend used for inference with Burn models.
    pub type BurnBackend = NdArray<f32>;
    /// CPU (NdArray) device used for inference with Burn models.
    pub type BurnDevice = NdArrayDevice;
}

pub use backend::{BurnBackend, BurnDevice};
