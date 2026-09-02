//! `refine_factor` vs the two service-level caches.
//!
//! A reranked and an unreranked query over the same vector are
//! different result sets, so they must not share cache entries:
//!
//! 1. **Exact cache** — `refine_factor` is part of the cache key.
//!    Repeating a request hits; changing only `refine_factor` misses.
//! 2. **Semantic sidecar** — the sidecar keys on k / nprobes /
//!    include_vector but not the rerank factor, so a request that
//!    carries `refine_factor` bypasses it entirely instead of
//!    reusing an unreranked neighbour's bytes.
//!
//! Runs on the local filesystem (embedded mode), so it needs no
//! MinIO and is not `#[ignore]`d.

use std::collections::HashMap;
use std::sync::Arc;

use firnflow_core::cache::NamespaceCache;
use firnflow_core::metrics::test_metrics;
use firnflow_core::{
    NamespaceId, NamespaceManager, NamespaceService, QueryCacheSource, QueryRequest,
    SemanticCacheRequest, StorageRoot, UpsertRow,
};

const DIM: usize = 8;

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[axis] = 1.0;
    v
}

/// Tilt slightly off-axis: cosine similarity to `unit_vector(axis)`
/// is sqrt(1 - drift^2), close to but below 1.0, so the request is
/// an exact-cache miss but a semantic-cache candidate.
fn near_unit_vector(axis: usize, drift: f32) -> Vec<f32> {
    let other = (axis + 1) % DIM;
    let mut v = vec![0.0_f32; DIM];
    v[axis] = (1.0_f32 - drift * drift).sqrt();
    v[other] = drift;
    v
}

fn request(vector: Vec<f32>, refine_factor: Option<u32>, semantic: bool) -> QueryRequest {
    QueryRequest {
        vector,
        vectors: None,
        k: 5,
        nprobes: None,
        refine_factor,
        text: None,
        filter: None,
        include_vector: false,
        semantic_cache: semantic.then_some(SemanticCacheRequest {
            enabled: true,
            min_similarity: Some(0.95),
        }),
        exact: false,
    }
}

async fn build_service(
    tmp: &tempfile::TempDir,
) -> (
    Arc<NamespaceService>,
    NamespaceId,
    Arc<firnflow_core::metrics::CoreMetrics>,
) {
    let metrics = test_metrics();
    let manager = Arc::new(NamespaceManager::new(
        StorageRoot::local(tmp.path().join("data")).unwrap(),
        HashMap::new(),
        Arc::clone(&metrics),
    ));
    let cache = Arc::new(
        NamespaceCache::new(
            16 * 1024 * 1024,
            &tmp.path().join("cache"),
            64 * 1024 * 1024,
            Arc::clone(&metrics),
        )
        .await
        .expect("build cache"),
    );
    let service = Arc::new(NamespaceService::new(
        Arc::clone(&manager),
        cache,
        Arc::clone(&metrics),
    ));
    let ns = NamespaceId::new("refine-cache").unwrap();
    let rows = (0..DIM)
        .map(|i| UpsertRow::from((i as u64, unit_vector(i))))
        .collect::<Vec<_>>();
    service.upsert(&ns, rows).await.expect("seed upsert");
    (service, ns, metrics)
}

#[tokio::test]
async fn reranked_and_unreranked_requests_have_separate_exact_cache_entries() {
    let tmp = tempfile::tempdir().unwrap();
    let (service, ns, _metrics) = build_service(&tmp).await;

    let plain = request(unit_vector(0), None, false);
    let refined = request(unit_vector(0), Some(2), false);

    // Populate and prove the exact cache works for the plain shape.
    let first = service
        .query_with_cache_source(&ns, &plain)
        .await
        .expect("plain #1");
    assert_eq!(first.cache_source, QueryCacheSource::Backend);
    let repeat = service
        .query_with_cache_source(&ns, &plain)
        .await
        .expect("plain #2");
    assert_eq!(repeat.cache_source, QueryCacheSource::ExactCache);

    // Same vector, same k — only refine_factor differs. Serving the
    // cached unreranked bytes here would silently drop the rerank,
    // so this must go to the backend.
    let refined_first = service
        .query_with_cache_source(&ns, &refined)
        .await
        .expect("refined #1");
    assert_eq!(
        refined_first.cache_source,
        QueryCacheSource::Backend,
        "a reranked request must not be served an unreranked cache entry"
    );

    // The refined shape caches under its own key.
    let refined_repeat = service
        .query_with_cache_source(&ns, &refined)
        .await
        .expect("refined #2");
    assert_eq!(refined_repeat.cache_source, QueryCacheSource::ExactCache);

    // And the refined entry does not leak back to plain requests.
    let plain_again = service
        .query_with_cache_source(&ns, &plain)
        .await
        .expect("plain #3");
    assert_eq!(plain_again.cache_source, QueryCacheSource::ExactCache);
    assert_eq!(plain_again.result, first.result);
}

#[tokio::test]
async fn a_reranked_request_bypasses_the_semantic_sidecar() {
    let tmp = tempfile::tempdir().unwrap();
    let (service, ns, metrics) = build_service(&tmp).await;

    // Seed the sidecar with an unreranked opt-in query.
    let seed = request(unit_vector(0), None, true);
    let seeded = service.query(&ns, &seed).await.expect("seed query");

    // Sanity: an unreranked near-duplicate is served by the sidecar.
    let near_plain = request(near_unit_vector(0, 0.05), None, true);
    let reused = service
        .query_with_cache_source(&ns, &near_plain)
        .await
        .expect("near-duplicate, no refine");
    assert_eq!(reused.cache_source, QueryCacheSource::SemanticCache);
    assert_eq!(reused.result.results, seeded.results);
    assert_eq!(metrics.semantic_cache_hits_value(&ns), 1);

    // The same near-duplicate carrying refine_factor must skip the
    // sidecar: its entries hold unreranked top-k bytes. The request
    // runs against the backend and ticks the shape-rejection counter.
    let near_refined = request(near_unit_vector(0, 0.06), Some(2), true);
    let refined = service
        .query_with_cache_source(&ns, &near_refined)
        .await
        .expect("near-duplicate, refined");
    assert_eq!(
        refined.cache_source,
        QueryCacheSource::Backend,
        "a reranked request must not reuse unreranked semantic-cache bytes"
    );
    assert_eq!(metrics.semantic_cache_hits_value(&ns), 1);
    assert_eq!(
        metrics.semantic_cache_rejections_value(&ns, "unsupported_query_shape"),
        1,
        "the bypass must be visible in the rejection counter"
    );
}
