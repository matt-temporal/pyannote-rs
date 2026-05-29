//! Facade that re-creates the parts of the `burn` umbrella crate this crate
//! uses, but assembled from burn's leaf sub-crates.
//!
//! We avoid depending on the `burn` umbrella crate because it declares
//! `burn-cpu` / `burn-wgpu` as optional dependencies; cargo's `links`-aware
//! resolver can spuriously activate `burn-cpu` (pulling `cubecl-cpu ->
//! liblzma`) when a downstream consumer also links lzma (e.g. ffmpeg-sidecar),
//! producing an unresolvable `links = "lzma"` conflict. Depending on the leaf
//! crates removes those optional deps as candidates.
//!
//! The generated model code refers to `burn::nn`, `burn::tensor`,
//! `burn::module`, `burn::prelude`, and `burn::backend::ndarray`. Each module
//! that needs these does `use crate::burn_facade as burn;`.

// burn_core re-exports module/tensor/prelude/config/record etc. at its root.
pub use burn_core::*;

/// Neural network building blocks (mirrors `burn::nn`).
pub mod nn {
    pub use burn_nn::*;
}

/// Backend implementations (mirrors `burn::backend`).
pub mod backend {
    /// NdArray (CPU) backend.
    pub mod ndarray {
        // Unused when the WGPU/Metal backend is active, but kept for the CPU path.
        #[allow(unused_imports)]
        pub use burn_ndarray::*;
    }
}
