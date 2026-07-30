//! Regression test for #103: a full-text or hybrid query against a
//! namespace that has rows but no BM25 index must answer 400 naming the
//! missing index, not 500.
//!
//! The distinction matters to clients, not just to tidiness. A 500 is
//! indistinguishable from a storage or IO failure, so a client with
//! retry logic burns its whole backoff budget on a request that cannot
//! succeed until an index is built. A 400 says so on the first attempt.
//!
//! Runs against a local-filesystem storage root, so unlike `api_fts.rs`
//! this needs no MinIO and is deliberately not `#[ignore]`d.

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use firnflow_api::router;
use serde_json::{Value, json};
use tower::ServiceExt;

mod common;
use common::{test_state_local, unique_namespace};

const DIM: usize = 8;

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[axis] = 1.0;
    v
}

async fn post_json(app: axum::Router, uri: String, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, json)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_query_without_fts_index_returns_400_naming_the_index() {
    let (state, _tmp) = test_state_local().await;
    let app = router(state);
    let ns = unique_namespace("fts-unindexed");

    let upsert_body = json!({
        "rows": [
            {"id": 1, "vector": unit_vector(0), "text": "the quick brown fox"},
            {"id": 2, "vector": unit_vector(1), "text": "a lazy dog sleeps"},
        ]
    });
    let (status, _) = post_json(app.clone(), format!("/ns/{ns}/upsert"), upsert_body).await;
    assert_eq!(status, StatusCode::OK, "upsert must succeed");

    // No fts-index call. This is the state a caller lands in after
    // their first write, which is exactly when they start querying.

    for (case, body) in [
        ("fts-only", json!({"text": "fox", "k": 2})),
        (
            "hybrid",
            json!({"vector": unit_vector(0), "text": "fox", "k": 2}),
        ),
        (
            "filtered fts",
            json!({"text": "fox", "k": 2, "filter": "id > 0"}),
        ),
    ] {
        let (status, body) = post_json(app.clone(), format!("/ns/{ns}/query"), body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{case}: must be a client error, got {status} {body}"
        );
        let msg = body["error"].as_str().unwrap_or_default();
        assert!(
            msg.contains("BM25 index") && msg.contains("/fts-index"),
            "{case}: error must name the index and the endpoint that builds it, got {msg:?}"
        );
    }

    // Vector-only on the same namespace is unaffected.
    let (status, body) = post_json(
        app.clone(),
        format!("/ns/{ns}/query"),
        json!({"vector": unit_vector(0), "k": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "vector-only must still succeed");
    assert_eq!(
        body["results"].as_array().map(|r| r.len()),
        Some(2),
        "vector-only must return both rows"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_query_on_unwritten_namespace_stays_200_and_empty() {
    // The trap described in #103: a namespace with no table answers
    // every query shape with an empty 200, so a fresh deployment looks
    // healthy until the first document lands.
    let (state, _tmp) = test_state_local().await;
    let app = router(state);
    let ns = unique_namespace("fts-unwritten");

    for (case, body) in [
        ("fts-only", json!({"text": "anything", "k": 2})),
        (
            "hybrid",
            json!({"vector": unit_vector(0), "text": "anything", "k": 2}),
        ),
        ("vector-only", json!({"vector": unit_vector(0), "k": 2})),
    ] {
        let (status, body) = post_json(app.clone(), format!("/ns/{ns}/query"), body).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{case}: unwritten namespace must not error, got {status} {body}"
        );
        assert_eq!(
            body["results"].as_array().map(|r| r.len()),
            Some(0),
            "{case}: unwritten namespace must return no hits"
        );
    }
}
