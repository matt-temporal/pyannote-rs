// `burn` facade assembled from leaf sub-crates (see burn_facade.rs for why).
pub(crate) mod burn_facade;

mod nn;

mod embedding;
mod identify;
mod segment;
mod wav;

pub use embedding::Embedding;
pub use embedding::EmbeddingExtractor;
pub use identify::EmbeddingManager;
pub use segment::{Segment, Segmenter, get_segments, get_segments_f32};
pub use wav::read_wav;
