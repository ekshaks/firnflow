//! Cacheability analysis for query filter predicates.
//!
//! The exact result cache keys on the full query parameters,
//! including the `filter` predicate's source text. That is correct
//! for a stable predicate like `section = 'warnings'`, whose meaning
//! does not change between two identical requests, and wrong for one
//! whose meaning does. `_ingested_at < now()` is the motivating
//! case: cached once, the same request keeps returning the same rows
//! long after the cutoff has moved, because the cache only turns over
//! when the namespace is next written to.
//!
//! This module decides, per request, whether a predicate is safe to
//! cache. A predicate that is not safe bypasses the exact cache in
//! both directions — it is neither read nor written — so it always
//! reflects the backend, at the cost of the cache's speedup.
//!
//! # Why the planner rather than the predicate text
//!
//! Matching on the source text does not work. `now` is a legal
//! column name, so a substring search flags predicates that are
//! perfectly stable. The reverse failure is worse: `CURRENT_TIMESTAMP`
//! carries no parentheses and parses as an identifier, not a call, so
//! a parser-level search for function names misses it entirely — it
//! only becomes a `now()` call after the query planner rewrites it.
//!
//! So the predicate is planned rather than scanned, using the same
//! planner and the same function registry that LanceDB uses to run
//! it. The resulting expression tree is walked for function calls and
//! each one is asked for its volatility.
//!
//! # Why `Immutable` and not "not `Volatile`"
//!
//! Volatility has three levels, and the obvious check is the wrong
//! one. `random()` and `uuid()` are `Volatile`: they can return a
//! different value for every row within a single query. But `now()`,
//! `current_date`, and `current_time` are `Stable` — fixed for the
//! duration of one query, and free to differ between queries. Fixed
//! within a query is exactly no help to a cache that spans queries,
//! so `Stable` is just as unsafe to cache as `Volatile`, and a check
//! for `Volatility::Volatile` alone would miss `now()` — the case
//! this module exists for.
//!
//! Only `Immutable` is cacheable.

use arrow_schema::SchemaRef;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_expr::{Expr, Volatility};
use lance_datafusion::planner::Planner;

/// Whether a filter predicate's result can be reused by a later
/// identical request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterCacheability {
    /// Every function in the predicate is `Immutable`, so the
    /// predicate means the same thing on every evaluation and its
    /// result set is safe to cache.
    Cacheable,
    /// The predicate calls a function whose result can change
    /// between two otherwise-identical requests. Carries the
    /// function's name for logging.
    Volatile {
        /// The first non-`Immutable` function found, by the name the
        /// planner resolved it to. `CURRENT_TIMESTAMP` reports as
        /// `now`, since that is what it plans to.
        function: String,
    },
    /// The predicate could not be planned against the namespace
    /// schema. Treated as uncacheable so an analysis failure can
    /// never cause a stale result; see [`classify_filter`].
    Unplannable,
}

impl FilterCacheability {
    /// Whether the exact result cache may be consulted and populated
    /// for this predicate.
    pub fn is_cacheable(&self) -> bool {
        matches!(self, Self::Cacheable)
    }
}

/// Classify `filter` against `schema` for exact-cache eligibility.
///
/// `schema` must be the namespace's Arrow schema; the planner
/// resolves column references against it, so passing an unrelated
/// schema turns every column reference into a planning failure.
///
/// A predicate that fails to plan returns
/// [`FilterCacheability::Unplannable`] rather than an error. Two
/// reasons to swallow it here: the same predicate is about to be
/// planned again by LanceDB against the same schema, so a genuine
/// syntax error surfaces there as a 400 with a message from the
/// engine that will actually run it, and duplicating that
/// classification in this module would mean two error paths to keep
/// in agreement. Reporting it as uncacheable is also the safe
/// direction — the worst case is a cache bypass on a request that is
/// about to fail anyway.
///
/// A panic in the planner is caught for the same reason it is caught
/// around query execution: Lance 6's SQL planner `todo!()`s on some
/// parseable-but-unimplemented syntax (national and bit string
/// literals, `N'x'` / `B'1'`), and a client-controlled predicate must
/// not be able to unwind the request task. Since this analysis runs
/// before the query does, it would otherwise hit that panic first,
/// ahead of the guard around execution.
pub fn classify_filter(schema: SchemaRef, filter: &str) -> FilterCacheability {
    let planned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Planner::new(schema).parse_filter(filter)
    }));
    match planned {
        Ok(Ok(expr)) => match first_non_immutable_function(&expr) {
            Some(function) => FilterCacheability::Volatile { function },
            None => FilterCacheability::Cacheable,
        },
        Ok(Err(_)) | Err(_) => FilterCacheability::Unplannable,
    }
}

/// Walk `expr` and return the name of the first function call whose
/// volatility is anything other than `Immutable`.
///
/// Only scalar functions are inspected. Aggregate and window
/// functions are not reachable here — the planner rejects both in a
/// filter predicate before this walk runs.
fn first_non_immutable_function(expr: &Expr) -> Option<String> {
    let mut found: Option<String> = None;
    // `Expr::apply` only returns an error if the closure does, and
    // this one cannot, so the result carries no information.
    let _ = expr.apply(|node| {
        if let Expr::ScalarFunction(call) = node {
            if !matches!(call.func.signature().volatility, Volatility::Immutable) {
                found = Some(call.func.name().to_string());
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use std::sync::Arc;

    /// Mirrors the namespace schema a filter is planned against:
    /// `id`, `text`, and the `_ingested_at` system column. The
    /// vector column is left out — it is not filterable and its type
    /// is irrelevant to predicate planning.
    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("text", DataType::Utf8, true),
            Field::new(
                "_ingested_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]))
    }

    fn classify(filter: &str) -> FilterCacheability {
        classify_filter(schema(), filter)
    }

    fn volatile_function(filter: &str) -> String {
        match classify(filter) {
            FilterCacheability::Volatile { function } => function,
            other => panic!("expected {filter:?} to be volatile, got {other:?}"),
        }
    }

    /// The common case: ordinary comparisons over columns and
    /// literals stay on the cached fast path.
    #[test]
    fn stable_predicates_are_cacheable() {
        for filter in [
            "id > 1000",
            "id > 1000 AND id < 2000",
            "text = 'warnings'",
            "text IS NOT NULL",
            "id IN (1, 2, 3)",
            "text LIKE 'warn%'",
            "_ingested_at < CAST('2026-01-01' AS TIMESTAMP)",
            "abs(CAST(id AS INT)) > 5",
        ] {
            assert_eq!(
                classify(filter),
                FilterCacheability::Cacheable,
                "expected {filter:?} to be cacheable"
            );
        }
    }

    /// `Stable` functions are the reason this module exists: fixed
    /// within one query, free to move between queries, and therefore
    /// unsafe to cache. A check for `Volatility::Volatile` alone
    /// would let every one of these through.
    #[test]
    fn stable_time_functions_are_not_cacheable() {
        assert_eq!(volatile_function("_ingested_at < now()"), "now");
        assert_eq!(
            volatile_function("_ingested_at < current_date"),
            "current_date"
        );
    }

    /// `CURRENT_TIMESTAMP` has no parentheses and is not a function
    /// call in the parsed SQL — it becomes one only after the
    /// planner rewrites it. This is the case a parser-level search
    /// for function names misses, so it is pinned separately.
    #[test]
    fn bare_current_timestamp_is_not_cacheable() {
        assert_eq!(volatile_function("_ingested_at < CURRENT_TIMESTAMP"), "now");
        assert_eq!(volatile_function("_ingested_at < current_timestamp"), "now");
    }

    /// `Volatile` functions, the case the obvious check does catch.
    #[test]
    fn volatile_functions_are_not_cacheable() {
        assert_eq!(volatile_function("random() < 0.5"), "random");
        assert_eq!(volatile_function("text != uuid()"), "uuid");
    }

    /// A volatile call anywhere in the tree taints the whole
    /// predicate, including under a boolean operator or nested as a
    /// function argument.
    #[test]
    fn volatility_is_detected_below_the_root() {
        assert_eq!(volatile_function("id > 0 AND random() < 0.5"), "random");
        assert_eq!(volatile_function("NOT (random() < 0.5)"), "random");
        assert_eq!(
            volatile_function("_ingested_at < date_trunc('day', now())"),
            "now"
        );
    }

    /// The issue's own trap: `now` is a legal column name. A
    /// predicate over a column called `now` is stable, and a
    /// substring search on the predicate text would wrongly flag it.
    #[test]
    fn column_named_now_is_still_cacheable() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("now", DataType::UInt64, false),
        ]));
        assert_eq!(
            classify_filter(schema, "now > 5"),
            FilterCacheability::Cacheable
        );
    }

    /// A predicate that cannot be planned is reported as
    /// uncacheable, not as an error and not as cacheable. LanceDB
    /// reports the actual syntax error to the caller.
    #[test]
    fn unplannable_predicates_are_not_cacheable() {
        for filter in ["id >", "!!!"] {
            assert_eq!(
                classify(filter),
                FilterCacheability::Unplannable,
                "expected {filter:?} to be unplannable"
            );
            assert!(!classify(filter).is_cacheable());
        }
    }

    /// Lance 6's planner `todo!()`s on national and bit string
    /// literals rather than returning an error. This analysis runs
    /// before the query does, so without a guard here a
    /// client-controlled predicate would unwind the request task
    /// ahead of the existing guard around query execution.
    #[test]
    fn planner_panics_are_caught_and_reported_unplannable() {
        for filter in ["text = N'x'", "text = B'1'"] {
            assert_eq!(
                classify(filter),
                FilterCacheability::Unplannable,
                "expected {filter:?} to be caught rather than panic"
            );
        }
    }

    /// The planner resolves column references lazily, so a predicate
    /// naming a column that does not exist still plans here and
    /// classifies as cacheable. That is harmless: the reference
    /// fails when LanceDB executes the query, which returns a 400
    /// and no result set, so nothing reaches the cache. Pinned
    /// because it is a surprising planner behaviour to rediscover.
    #[test]
    fn unknown_column_plans_and_is_left_to_the_engine() {
        assert_eq!(
            classify("no_such_column = 1"),
            FilterCacheability::Cacheable
        );
    }
}
