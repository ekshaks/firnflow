//! Query filter behavior through `NamespaceService` on local storage.
//!
//! Covers exact-cache splitting by filter and semantic-cache rejection
//! for filtered requests without requiring MinIO.

use std::collections::HashMap;
use std::sync::Arc;

use firnflow_core::cache::NamespaceCache;
use firnflow_core::metrics::test_metrics;
use firnflow_core::{
    FirnflowError, NamespaceId, NamespaceManager, NamespaceService, QueryCacheSource, QueryRequest,
    SemanticCacheRequest, StorageRoot, UpsertRow,
};
use tempfile::TempDir;

const DIM: usize = 8;

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[axis] = 1.0;
    v
}

fn request(filter: Option<&str>) -> QueryRequest {
    QueryRequest {
        vector: unit_vector(0),
        vectors: None,
        k: 10,
        nprobes: None,
        text: None,
        filter: filter.map(str::to_string),
        include_vector: false,
        semantic_cache: None,
    }
}

async fn local_service() -> (NamespaceService, NamespaceId, TempDir, TempDir) {
    let dir = TempDir::new().unwrap();
    let cache_dir = TempDir::new().unwrap();
    let metrics = test_metrics();
    let manager = Arc::new(NamespaceManager::new(
        StorageRoot::local(dir.path()).unwrap(),
        HashMap::new(),
        Arc::clone(&metrics),
    ));
    let cache = Arc::new(
        NamespaceCache::new(
            16 * 1024 * 1024,
            cache_dir.path(),
            64 * 1024 * 1024,
            Arc::clone(&metrics),
        )
        .await
        .expect("cache"),
    );
    let service = NamespaceService::new(Arc::clone(&manager), cache, metrics);
    let ns = NamespaceId::new("service-query-filter").unwrap();

    let rows: Vec<UpsertRow> = vec![
        (1u64, unit_vector(0)).into(),
        (2u64, unit_vector(1)).into(),
        (3u64, unit_vector(2)).into(),
    ];
    service.upsert(&ns, rows).await.expect("seed upsert");

    (service, ns, dir, cache_dir)
}

/// A filtered query whose predicate is the caller's fault must map to
/// `InvalidRequest` (400), not `Backend` (500). Covers a spread of predicate
/// failure shapes: a SQL syntax error, an unknown column, a type mismatch, an
/// unknown function, and an unsupported operator. The last two are the cases a
/// narrower message-matching classifier mislabeled as 500, so they pin the
/// broad classification. Genuine backend failures on a filtered query still map
/// to `Backend`; that path is not reachable from local storage, so it is
/// covered by the classifier's default arm rather than a test here.
#[tokio::test]
async fn filtered_predicate_errors_map_to_invalid_request() {
    let (service, ns, _dir, _cache_dir) = local_service().await;
    for bad in [
        "id =",               // SQL parse error
        "nope > 1",           // unknown column
        "text > 1",           // type mismatch
        "no_such_fn(id) = 1", // unknown function
        "id @> 1",            // unsupported operator
    ] {
        let req = request(Some(bad));
        let err = service
            .query_with_cache_source(&ns, &req)
            .await
            .expect_err("malformed predicate should error");
        match err {
            FirnflowError::InvalidRequest(msg) => {
                assert!(msg.contains("filter"), "predicate {bad:?}: {msg}")
            }
            other => panic!("predicate {bad:?}: expected InvalidRequest, got {other:?}"),
        }
    }
}

/// A predicate that parses but reaches an unimplemented path in Lance's SQL
/// planner (national or bit string literals) panics inside `execute()`. The
/// filtered path must catch that and report a 400, not unwind the request.
#[tokio::test]
async fn filtered_unsupported_syntax_maps_to_invalid_request() {
    let (service, ns, _dir, _cache_dir) = local_service().await;
    for bad in ["text = N'x'", "text = B'1'"] {
        let req = request(Some(bad));
        let err = service
            .query_with_cache_source(&ns, &req)
            .await
            .expect_err("unsupported predicate syntax should error, not panic");
        match err {
            FirnflowError::InvalidRequest(msg) => {
                assert!(msg.contains("filter"), "predicate {bad:?}: {msg}")
            }
            other => panic!("predicate {bad:?}: expected InvalidRequest, got {other:?}"),
        }
    }
}

/// A filtered full-text query on a namespace with no inverted index fails with
/// `InvalidInput`. The filter-error classifier is deliberately broad, so this
/// maps to a 400 rather than a 500. That is an accepted trade (a missing-index
/// 400 reads as "build the index"); the alternative, message-matching to force
/// it to 500, would mislabel genuine bad predicates as backend errors. This
/// test pins the chosen behaviour so a future change to it is a conscious one.
#[tokio::test]
async fn filtered_fts_without_index_maps_to_invalid_request() {
    let (service, ns, _dir, _cache_dir) = local_service().await;
    let req = QueryRequest {
        vector: Vec::new(),
        vectors: None,
        k: 10,
        nprobes: None,
        text: Some("anything".into()),
        filter: Some("id > 1".into()),
        include_vector: false,
        semantic_cache: None,
    };
    let err = service
        .query_with_cache_source(&ns, &req)
        .await
        .expect_err("fts query without an index should error");
    assert!(
        matches!(err, FirnflowError::InvalidRequest(_)),
        "missing FTS index on a filtered query maps to InvalidRequest, got {err:?}"
    );
}

#[tokio::test]
async fn filtered_and_unfiltered_queries_cache_independently() {
    let (service, ns, _dir, _cache_dir) = local_service().await;

    let unfiltered = request(None);
    let filtered = request(Some("id > 1"));

    let a = service
        .query_with_cache_source(&ns, &unfiltered)
        .await
        .expect("unfiltered #1");
    assert_eq!(a.cache_source, QueryCacheSource::Backend);
    let mut ids_a: Vec<u64> = a.result.results.iter().map(|r| r.id).collect();
    ids_a.sort_unstable();
    assert_eq!(ids_a, vec![1, 2, 3]);

    let b = service
        .query_with_cache_source(&ns, &filtered)
        .await
        .expect("filtered #1");
    assert_eq!(b.cache_source, QueryCacheSource::Backend);
    let mut ids_b: Vec<u64> = b.result.results.iter().map(|r| r.id).collect();
    ids_b.sort_unstable();
    assert_eq!(ids_b, vec![2, 3]);

    let a2 = service
        .query_with_cache_source(&ns, &unfiltered)
        .await
        .expect("unfiltered #2");
    assert_eq!(a2.cache_source, QueryCacheSource::ExactCache);
    assert_eq!(a2.result, a.result);

    let b2 = service
        .query_with_cache_source(&ns, &filtered)
        .await
        .expect("filtered #2");
    assert_eq!(b2.cache_source, QueryCacheSource::ExactCache);
    assert_eq!(b2.result, b.result);
}

#[tokio::test]
async fn distinct_filters_do_not_collide_in_exact_cache() {
    let (service, ns, _dir, _cache_dir) = local_service().await;

    let lt = request(Some("id < 3"));
    let gt = request(Some("id > 1"));

    let a = service
        .query_with_cache_source(&ns, &lt)
        .await
        .expect("lt filter");
    assert_eq!(a.cache_source, QueryCacheSource::Backend);
    let mut ids_a: Vec<u64> = a.result.results.iter().map(|r| r.id).collect();
    ids_a.sort_unstable();
    assert_eq!(ids_a, vec![1, 2]);

    let b = service
        .query_with_cache_source(&ns, &gt)
        .await
        .expect("gt filter");
    assert_eq!(b.cache_source, QueryCacheSource::Backend);
    let mut ids_b: Vec<u64> = b.result.results.iter().map(|r| r.id).collect();
    ids_b.sort_unstable();
    assert_eq!(ids_b, vec![2, 3]);
}

#[tokio::test]
async fn filtered_semantic_cache_request_is_rejected() {
    let (service, ns, _dir, _cache_dir) = local_service().await;
    let mut req = request(Some("id > 1"));
    req.semantic_cache = Some(SemanticCacheRequest {
        enabled: true,
        min_similarity: None,
    });

    let err = service
        .query_with_cache_source(&ns, &req)
        .await
        .expect_err("filtered semantic-cache query should reject");
    match err {
        FirnflowError::InvalidRequest(msg) => assert!(msg.contains("filter"), "{msg}"),
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}
