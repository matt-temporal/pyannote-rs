// Offline diarization tuning harness.
//
// Reads a 16kHz mono wav, runs pyannote segmentation + embedding, then:
//   (a) sweeps the agglomerative-clustering cosine-distance threshold (legacy
//       baseline), and
//   (b) runs normalized spectral clustering with auto-k via the eigengap
//       heuristic (the new, threshold-free approach).
// A companion Python script maps segments to ground-truth speakers (by
// timestamp) and scores each assignment line.
//
//   cargo run --release --example tune <audio.wav>
//
// Output format (stdout):
//   SEG <idx> <start_sec> <end_sec>
//   ...
//   THRESH <t> <id,id,id,...>            (one line per swept threshold)
//   THRESH 0.00 <id,id,...>              (sentinel line carrying the SPECTRAL ids)
//   SPECTRAL k=<k> <id,id,id,...>        (human-readable spectral line)
// Eigenvalue/gap diagnostics go to stderr.

use anyhow::Result;
use nalgebra::{DMatrix, SymmetricEigen};
use pyannote_rs::{EmbeddingExtractor, Segmenter};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = na.sqrt() * nb.sqrt();
    if d == 0.0 { 0.0 } else { dot / d }
}

/// Average-linkage agglomerative clustering over a precomputed distance matrix.
/// Returns a speaker id (1-based) per segment, ids ordered by earliest segment.
fn cluster(dist: &[Vec<f32>], n: usize, threshold: f32) -> Vec<u32> {
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    loop {
        if clusters.len() < 2 {
            break;
        }
        let mut best = (0usize, 1usize);
        let mut best_d = f32::MAX;
        for a in 0..clusters.len() {
            for b in (a + 1)..clusters.len() {
                let mut sum = 0.0f32;
                let mut cnt = 0usize;
                for &i in &clusters[a] {
                    for &j in &clusters[b] {
                        sum += dist[i][j];
                        cnt += 1;
                    }
                }
                let avg = sum / cnt as f32;
                if avg < best_d {
                    best_d = avg;
                    best = (a, b);
                }
            }
        }
        if best_d > threshold {
            break;
        }
        let (a, b) = best;
        let moved = std::mem::take(&mut clusters[b]);
        clusters[a].extend(moved);
        clusters.remove(b);
    }
    order_ids(&clusters, n, |i| i)
}

/// Order clusters by their earliest member key and produce a 1-based id per item.
fn order_ids<F: Fn(usize) -> usize>(clusters: &[Vec<usize>], n: usize, _k: F) -> Vec<u32> {
    let mut ordered: Vec<(usize, &Vec<usize>)> =
        clusters.iter().map(|m| (*m.iter().min().unwrap(), m)).collect();
    ordered.sort_by_key(|(e, _)| *e);
    let mut out = vec![0u32; n];
    for (ci, (_, members)) in ordered.iter().enumerate() {
        for &i in members.iter() {
            out[i] = (ci + 1) as u32;
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Spectral clustering with eigengap-based auto-k.
// ----------------------------------------------------------------------------

/// Tunable: affinity row-pruning fraction. For each row we keep the top
/// `AFFINITY_PRUNE_P` fraction of off-diagonal similarities and zero the rest,
/// then symmetrize A = max(A, A^T). This is the row-wise refinement from the
/// Google "Speaker Diarization with LSTM" pipeline; it suppresses spurious
/// cross-speaker affinity and sharpens the spectral gap. 1.0 == no pruning.
const AFFINITY_PRUNE_P_DEFAULT: f64 = 0.30;

/// Tunable: maximum number of speakers the eigengap search will consider.
const K_MAX: usize = 10;

fn affinity_prune_p() -> f64 {
    std::env::var("PRUNE_P")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(AFFINITY_PRUNE_P_DEFAULT)
}

/// Result of a spectral run, for logging and scoring.
struct SpectralResult {
    k: usize,
    ids: Vec<u32>,
    eigenvalues: Vec<f64>,
    gap: f64,
}

/// Normalized spectral clustering over the cosine-similarity matrix `sim`
/// (sim[i][j] in [-1,1]). `starts` is used only to order final ids by earliest
/// speech time. Returns chosen k, per-item 1-based ids, smallest eigenvalues,
/// and the winning eigengap. Never panics.
fn spectral_cluster(sim: &[Vec<f32>], starts: &[f64]) -> SpectralResult {
    let prune_p = affinity_prune_p();
    let n = sim.len();
    if n == 0 {
        return SpectralResult { k: 0, ids: vec![], eigenvalues: vec![], gap: 0.0 };
    }
    if n == 1 {
        return SpectralResult { k: 1, ids: vec![1], eigenvalues: vec![0.0], gap: 0.0 };
    }

    // 1. Build affinity A: clamp negative cosines to 0, zero diagonal.
    let mut a = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            a[i][j] = (sim[i][j] as f64).max(0.0);
        }
    }

    // 1b. Row-wise pruning: keep top-p fraction of each row, zero the rest.
    if prune_p < 1.0 {
        for i in 0..n {
            let mut vals: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| a[i][j]).collect();
            vals.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
            let keep = ((n - 1) as f64 * prune_p).ceil() as usize;
            let keep = keep.max(1).min(vals.len());
            let cutoff = vals[keep - 1];
            for j in 0..n {
                if j != i && a[i][j] < cutoff {
                    a[i][j] = 0.0;
                }
            }
        }
        // Symmetrize: A = max(A, A^T).
        for i in 0..n {
            for j in (i + 1)..n {
                let m = a[i][j].max(a[j][i]);
                a[i][j] = m;
                a[j][i] = m;
            }
        }
    }

    // 2. Normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
    let mut dinv = vec![0.0f64; n];
    for i in 0..n {
        let deg: f64 = a[i].iter().sum();
        dinv[i] = if deg > 1e-12 { 1.0 / deg.sqrt() } else { 0.0 };
    }
    let mut lsym = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let norm = dinv[i] * a[i][j] * dinv[j];
            lsym[(i, j)] = if i == j { 1.0 - norm } else { -norm };
        }
    }
    // Symmetrize numerically (guard against tiny asymmetry).
    for i in 0..n {
        for j in (i + 1)..n {
            let m = 0.5 * (lsym[(i, j)] + lsym[(j, i)]);
            lsym[(i, j)] = m;
            lsym[(j, i)] = m;
        }
    }

    // 3. Symmetric eigendecomposition; sort ascending.
    let eig = SymmetricEigen::new(lsym);
    let mut pairs: Vec<(f64, usize)> =
        eig.eigenvalues.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_vals: Vec<f64> = pairs.iter().map(|p| p.0).collect();

    // 4. Eigengap heuristic: k = argmax_k (lambda_{k+1} - lambda_k), k in [1,K_MAX].
    // 4. Eigengap heuristic. The first gap (lambda_1 - lambda_0) corresponds to
    //    k=1: for a connected graph lambda_0 == 0 and that gap is ALWAYS large,
    //    so a naive argmax always returns k=1 and never detects >1 speaker.
    //    We therefore search the largest gap over k>=2 and only fall back to
    //    k=1 when no such gap exists (n<=2).
    let kmax = K_MAX.min(n);
    let mut best_k = 1usize;
    let mut best_gap = f64::MIN;
    for k in 2..kmax {
        let gap = sorted_vals[k] - sorted_vals[k - 1];
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    if best_gap == f64::MIN {
        // n<=2: no k>=2 candidate. Decide 1 vs 2 from the single gap available.
        best_gap = if kmax >= 2 { sorted_vals[1] - sorted_vals[0] } else { 0.0 };
        best_k = 1;
    }
    let k = best_k.max(1);

    let smallest: Vec<f64> = sorted_vals.iter().take(8).cloned().collect();
    let gaps: Vec<f64> = (1..kmax).map(|k| sorted_vals[k] - sorted_vals[k - 1]).collect();
    eprintln!(
        "  eigengap candidates (gap before k clusters): {:?}",
        gaps.iter().enumerate().map(|(i, g)| (i + 1, (g * 10000.0).round() / 10000.0)).collect::<Vec<_>>()
    );

    if k <= 1 {
        return SpectralResult {
            k: 1,
            ids: vec![1u32; n],
            eigenvalues: smallest,
            gap: best_gap,
        };
    }

    // 5. Spectral embedding: first k eigenvectors (smallest eigenvalues),
    //    row-normalized (L2).
    let mut feat = vec![vec![0.0f64; k]; n];
    for (col, &(_, orig)) in pairs.iter().take(k).enumerate() {
        let v = eig.eigenvectors.column(orig);
        for i in 0..n {
            feat[i][col] = v[i];
        }
    }
    for row in feat.iter_mut() {
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    }

    // 6. k-means (Lloyd's) with restarts.
    let labels = kmeans(&feat, k, 10, 100);

    // 7. Order ids by earliest start_time.
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in labels.iter().enumerate() {
        clusters[c].push(i);
    }
    clusters.retain(|c| !c.is_empty());
    let ids = order_ids_by_time(&clusters, n, starts);

    SpectralResult { k: ids.iter().cloned().collect::<std::collections::HashSet<_>>().len(), ids, eigenvalues: smallest, gap: best_gap }
}

/// Order clusters by earliest start_time and emit 1-based ids.
fn order_ids_by_time(clusters: &[Vec<usize>], n: usize, starts: &[f64]) -> Vec<u32> {
    let mut ordered: Vec<(f64, &Vec<usize>)> = clusters
        .iter()
        .map(|m| (m.iter().map(|&i| starts[i]).fold(f64::MAX, f64::min), m))
        .collect();
    ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0u32; n];
    for (ci, (_, members)) in ordered.iter().enumerate() {
        for &i in members.iter() {
            out[i] = (ci + 1) as u32;
        }
    }
    out
}

/// Simple Lloyd's k-means with `restarts` random inits and `max_iter` cap.
/// Returns a cluster index in [0,k) per row. Deterministic seed for repeatability.
fn kmeans(data: &[Vec<f64>], k: usize, restarts: usize, max_iter: usize) -> Vec<usize> {
    let n = data.len();
    let dim = data[0].len();
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let dist2 = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    };

    let mut best_labels = vec![0usize; n];
    let mut best_inertia = f64::MAX;

    for _ in 0..restarts {
        // k-means++ style seeding (lightweight): first center random, rest by
        // squared-distance weighting.
        let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
        centers.push(data[(next() as usize) % n].clone());
        while centers.len() < k {
            let mut weights = vec![0.0f64; n];
            for (i, p) in data.iter().enumerate() {
                weights[i] = centers
                    .iter()
                    .map(|c| dist2(p, c))
                    .fold(f64::MAX, f64::min);
            }
            let total: f64 = weights.iter().sum();
            if total <= 1e-12 {
                centers.push(data[(next() as usize) % n].clone());
                continue;
            }
            let mut target = (next() as f64 / u64::MAX as f64) * total;
            let mut chosen = n - 1;
            for (i, &w) in weights.iter().enumerate() {
                target -= w;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centers.push(data[chosen].clone());
        }

        let mut labels = vec![0usize; n];
        for _ in 0..max_iter {
            let mut changed = false;
            for (i, p) in data.iter().enumerate() {
                let mut best = 0usize;
                let mut bd = f64::MAX;
                for (c, ctr) in centers.iter().enumerate() {
                    let d = dist2(p, ctr);
                    if d < bd {
                        bd = d;
                        best = c;
                    }
                }
                if labels[i] != best {
                    labels[i] = best;
                    changed = true;
                }
            }
            // Recompute centers.
            let mut sums = vec![vec![0.0f64; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, p) in data.iter().enumerate() {
                let c = labels[i];
                counts[c] += 1;
                for d in 0..dim {
                    sums[c][d] += p[d];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        centers[c][d] = sums[c][d] / counts[c] as f64;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let inertia: f64 = data
            .iter()
            .enumerate()
            .map(|(i, p)| dist2(p, &centers[labels[i]]))
            .sum();
        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = labels;
        }
    }
    best_labels
}

fn main() -> Result<()> {
    let audio_path = std::env::args().nth(1).expect("usage: tune <audio.wav>");
    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path)?;
    eprintln!("read_wav: {} samples @ {}Hz ({:.1} min)", samples.len(), sample_rate, samples.len() as f64 / sample_rate as f64 / 60.0);
    let extractor = EmbeddingExtractor::new("src/nn/speaker_identification/model.bpk")?;
    let segmenter = Segmenter::new("src/nn/segmentation/model.bpk")?;

    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut embs: Vec<Vec<f32>> = Vec::new();
    let (mut raw, mut short, mut extract_err) = (0usize, 0usize, 0usize);

    for seg in segmenter.iter_segments(&samples, sample_rate)? {
        let seg = match seg { Ok(s) => s, Err(e) => { eprintln!("seg err: {}", e); continue; } };
        raw += 1;
        // skip very short segments (unreliable embeddings)
        if seg.samples.len() < 16000 / 2 {
            short += 1;
            continue;
        }
        match extractor.extract(&seg.samples, sample_rate) {
            Ok(e) => {
                starts.push(seg.start);
                ends.push(seg.end);
                embs.push(e.as_slice().to_vec());
            }
            Err(e) => { extract_err += 1; if extract_err <= 3 { eprintln!("extract err: {}", e); } }
        }
    }
    eprintln!("raw segments: {}, dropped short: {}, extract errors: {}", raw, short, extract_err);

    let n = embs.len();
    eprintln!("segments: {}", n);
    for i in 0..n {
        println!("SEG {} {:.2} {:.2}", i, starts[i], ends[i]);
    }

    // similarity + distance matrices
    let mut sim = vec![vec![0.0f32; n]; n];
    let mut dist = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        sim[i][i] = 1.0;
        for j in (i + 1)..n {
            let c = cosine(&embs[i], &embs[j]);
            sim[i][j] = c;
            sim[j][i] = c;
            let d = 1.0 - c;
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Optionally dump the similarity matrix + segment times for fast offline
    // spectral-parameter prototyping in Python (set SIM_DUMP=/path).
    if let Ok(path) = std::env::var("SIM_DUMP") {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("create SIM_DUMP");
        for i in 0..n {
            writeln!(f, "START {} {:.3} {:.3}", i, starts[i], ends[i]).unwrap();
        }
        for i in 0..n {
            let row: Vec<String> = (0..n).map(|j| format!("{:.5}", sim[i][j])).collect();
            writeln!(f, "SIM {}", row.join(" ")).unwrap();
        }
        eprintln!("dumped similarity matrix to {}", path);
    }

    let mut t = 0.30f32;
    while t <= 0.95001 {
        let ids = cluster(&dist, n, t);
        let line: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
        println!("THRESH {:.2} {}", t, line.join(","));
        t += 0.025;
    }

    // Spectral clustering with auto-k (the new threshold-free approach).
    let sr = spectral_cluster(&sim, &starts);
    eprintln!(
        "SPECTRAL: prune_p={} K_MAX={} | chosen k={} (eigengap={:.4}) | smallest eigenvalues={:?}",
        affinity_prune_p(),
        K_MAX,
        sr.k,
        sr.gap,
        sr.eigenvalues.iter().map(|v| (v * 10000.0).round() / 10000.0).collect::<Vec<f64>>(),
    );
    let sline: Vec<String> = sr.ids.iter().map(|x| x.to_string()).collect();
    println!("SPECTRAL k={} {}", sr.k, sline.join(","));
    // Sentinel THRESH line so the existing scorer picks up the spectral result.
    println!("THRESH 0.00 {}", sline.join(","));

    Ok(())
}
