//! End-to-end tests for `refine_factor` on the query path.
//!
//! An IVF_PQ index stores a lossy product-quantised (PQ) sketch of
//! every vector and ranks candidates by distance to the sketch, not
//! to the vector itself. `refine_factor: n` makes Lance pull `n * k`
//! candidates by sketch distance and re-score them against the stored
//! full-precision vectors before returning the top `k`.
//!
//! The correctness anchor used here: when `n * k` is at least the
//! table's row count and every partition is probed, the candidate
//! pool is the whole table, so the re-scoring pass degenerates into
//! an exact search. The refined top-k must then equal the exact
//! top-k the test computes on the CPU from the same rows — no
//! statistical tolerance, no flakiness from PQ training randomness.
//!
//! Runs on the local filesystem (embedded mode), so it needs no
//! MinIO and is not `#[ignore]`d.

use std::collections::HashMap;
use std::collections::HashSet;

use firnflow_core::metrics::test_metrics;
use firnflow_core::{
    FirnflowError, NamespaceId, NamespaceManager, StorageRoot, UpsertRow, VectorKind,
};
use tempfile::TempDir;

/// Wide enough that PQ (8 sub-vectors of 8 dims each) is genuinely
/// lossy on random data; small enough that the test stays fast.
const DIM: usize = 64;
/// 2,000 rows: above Lance's 256-row PQ training floor, and small
/// enough that `refine_factor = 200` with `k = 10` covers the whole
/// table (200 * 10 = 2,000 candidates).
const ROWS: u64 = 2_000;
const K: usize = 10;
/// Covers the full table: REFINE_ALL * K >= ROWS.
const REFINE_ALL: u32 = 200;

fn local_manager(dir: &TempDir) -> NamespaceManager {
    NamespaceManager::new(
        StorageRoot::local(dir.path()).unwrap(),
        HashMap::new(),
        test_metrics(),
    )
}

/// Deterministic pseudo-random vector for row `id`. A fixed-seed
/// SplitMix64 keeps the dataset identical across runs, so the exact
/// top-k computed below is stable. (PQ training inside Lance may
/// still be seeded randomly; the full-table-refine anchor is immune
/// to that.)
fn row_vector(id: u64) -> Vec<f32> {
    let mut state = id.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..DIM)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Map to [-1, 1).
            (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// The query vector. Offset far past every row id so it never
/// coincides with a stored vector.
fn query_vector() -> Vec<f32> {
    row_vector(1_000_000)
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Exact top-k ids by brute-force L2 over the same generated rows.
fn exact_top_k(query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = (0..ROWS)
        .map(|id| (id, l2(&row_vector(id), query)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Seed the namespace and build a one-partition IVF_PQ index.
/// One partition means the default probe count already scans every
/// partition, so any recall the index loses is lost to PQ scoring
/// alone — the effect `refine_factor` exists to undo.
async fn seeded_indexed(dir: &TempDir) -> (NamespaceManager, NamespaceId) {
    let manager = local_manager(dir);
    let ns = NamespaceId::new("refine-factor").unwrap();
    let rows: Vec<UpsertRow> = (0..ROWS).map(|id| (id, row_vector(id)).into()).collect();
    manager.upsert(&ns, rows).await.expect("seed upsert");
    manager
        .create_index(&ns, Some(1), Some(8), None)
        .await
        .expect("build IVF_PQ index");
    (manager, ns)
}

fn ids(results: &firnflow_core::QueryResultSet) -> Vec<u64> {
    results.results.iter().map(|r| r.id).collect()
}

#[tokio::test]
async fn full_table_refine_returns_the_exact_top_k() {
    let dir = TempDir::new().unwrap();
    let (manager, ns) = seeded_indexed(&dir).await;
    let truth: HashSet<u64> = exact_top_k(&query_vector(), K).into_iter().collect();

    // Unrefined: whatever the quantiser ranked highest.
    let unrefined = manager
        .query(
            &ns,
            query_vector(),
            None,
            K,
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .await
        .expect("unrefined query");
    let unrefined_overlap = ids(&unrefined)
        .iter()
        .filter(|id| truth.contains(id))
        .count();

    // Refined over a candidate pool that covers the whole table:
    // must be the exact answer, not an approximation of it.
    let refined = manager
        .query(
            &ns,
            query_vector(),
            None,
            K,
            None,
            Some(REFINE_ALL),
            None,
            None,
            false,
            false,
        )
        .await
        .expect("refined query");
    let refined_ids: HashSet<u64> = ids(&refined).into_iter().collect();

    assert_eq!(
        refined_ids, truth,
        "refine over the full table must reproduce the exact top-{K}"
    );
    // The assertion above only means something while the quantiser is
    // losing neighbours on this fixture. If the unrefined query were
    // already exact, deleting the refine_factor wiring would leave
    // the test green. 8 sub-vectors over 64 random dimensions loses
    // most of the top ten, so this holds with a wide margin.
    assert!(
        unrefined_overlap < K,
        "the unrefined query already returned the exact top-{K}, so this \
         fixture no longer exercises quantisation loss and the equality \
         above would pass with refinement switched off"
    );
}

#[tokio::test]
async fn refine_works_with_vector_projection_off() {
    // `include_vector: false` projects the vector column out of the
    // *result set*. The rerank pass reads stored vectors during the
    // scan, before that projection, so refinement must still return
    // the exact answer. This pins the interaction down.
    let dir = TempDir::new().unwrap();
    let (manager, ns) = seeded_indexed(&dir).await;
    let truth: HashSet<u64> = exact_top_k(&query_vector(), K).into_iter().collect();

    let refined_no_vec = manager
        .query(
            &ns,
            query_vector(),
            None,
            K,
            None,
            Some(REFINE_ALL),
            None,
            None,
            false,
            false,
        )
        .await
        .expect("refined query with include_vector=false");
    let got: HashSet<u64> = ids(&refined_no_vec).into_iter().collect();
    assert_eq!(got, truth, "projection must not disturb the rerank");
    assert!(
        refined_no_vec.results.iter().all(|r| r.vector.is_none()),
        "include_vector=false must still omit vectors from the response"
    );
}

#[tokio::test]
async fn refine_factor_zero_is_rejected_before_any_search() {
    let dir = TempDir::new().unwrap();
    let manager = local_manager(&dir);
    let ns = NamespaceId::new("refine-zero").unwrap();
    let rows: Vec<UpsertRow> = (0..4u64).map(|id| (id, row_vector(id)).into()).collect();
    manager.upsert(&ns, rows).await.expect("seed upsert");

    let err = manager
        .query(
            &ns,
            row_vector(0),
            None,
            3,
            None,
            Some(0),
            None,
            None,
            false,
            false,
        )
        .await
        .expect_err("refine_factor: 0 must be a bad request");
    assert!(
        matches!(err, FirnflowError::InvalidRequest(_)),
        "expected InvalidRequest, got: {err}"
    );
    assert!(
        format!("{err}").contains("refine_factor"),
        "the message must name the offending field: {err}"
    );
}

#[tokio::test]
async fn refine_reaches_a_multivector_namespace() {
    // `refine_factor` is applied after the single/multi shape match,
    // so a multivector (late-interaction MaxSim) query carries it
    // into Lance too. This pins down that the combination works
    // rather than erroring inside the engine.
    const SUB_DIM: usize = 8;
    fn unit(axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; SUB_DIM];
        v[axis] = 1.0;
        v
    }
    fn multi(id: u64, subs: Vec<Vec<f32>>) -> UpsertRow {
        UpsertRow {
            id,
            vector: Vec::new(),
            vectors: Some(subs),
            text: None,
            attributes: Default::default(),
        }
    }

    let dir = TempDir::new().unwrap();
    let manager = local_manager(&dir);
    let ns = NamespaceId::new("refine-multivector").unwrap();

    // Row 1 matches the query on both sub-vectors; every other row
    // lives on axes 2..8, so under cosine MaxSim row 1 is the unique
    // best hit and the assertion is tie-free. 300 single-sub-vector
    // fillers push the sub-vector count past Lance's 256 PQ training
    // floor.
    let mut rows = vec![
        multi(1, vec![unit(0), unit(1)]),
        multi(2, vec![unit(2), unit(3)]),
    ];
    for i in 0..300u64 {
        rows.push(multi(100 + i, vec![unit(2 + (i as usize) % (SUB_DIM - 2))]));
    }
    manager.upsert(&ns, rows).await.expect("seed upsert");
    assert_eq!(manager.kind_for(&ns), Some(VectorKind::Multivector));
    manager
        .create_index(&ns, Some(4), Some(2), None)
        .await
        .expect("build multivector IVF_PQ index");

    let unrefined = manager
        .query(
            &ns,
            Vec::new(),
            Some(vec![unit(0), unit(1)]),
            3,
            Some(4),
            None,
            None,
            None,
            false,
            false,
        )
        .await
        .expect("unrefined multivector query");
    let refined = manager
        .query(
            &ns,
            Vec::new(),
            Some(vec![unit(0), unit(1)]),
            3,
            Some(4),
            Some(4),
            None,
            None,
            false,
            false,
        )
        .await
        .expect("refined multivector query");
    let unrefined_top = unrefined.results.first().expect("an unrefined hit");
    let refined_top = refined.results.first().expect("a refined hit");
    assert_eq!(
        refined_top.id, 1,
        "the row matching both sub-vectors must stay on top under refinement"
    );
    // Refinement must not change *which* row wins, only how the
    // candidates were scored.
    assert_eq!(unrefined_top.id, refined_top.id);
    // Ranking alone cannot tell the two plans apart here, because row
    // 1 wins either way. The score can: Lance reports the refined
    // multivector plan on a different scale from the unrefined one, a
    // negated similarity sum rather than a MaxSim distance. Deleting
    // the `vq.refine_factor(rf)` call makes the two scores identical
    // and fails this line, which is the point of asserting on it.
    assert_ne!(
        unrefined_top.score, refined_top.score,
        "refined and unrefined scores are identical ({}), so refine_factor \
         never reached the multivector plan",
        refined_top.score
    );
}
