//! Regression test for #104: a startup failure to open the result
//! cache must name the directory and the variable that selected it.
//!
//! The reported symptom was a container refusing to start under
//! `readOnlyRootFilesystem: true` with nothing but `Read-only file
//! system (os error 30)`. Neither the path nor the environment variable
//! appeared, so there was nothing to search for and no way to tell
//! which of the two cache directories was at fault. The obvious next
//! move is to relax the security context until it starts, which works
//! and leaves a worse deployment behind.
//!
//! Needs no MinIO: `build_state` constructs the manager lazily and
//! fails on local filesystem work well before any object-store call.

use std::collections::HashMap;
use std::path::PathBuf;

use firnflow_api::build_state;
use firnflow_api::config::AppConfig;
use firnflow_api::rate_limit::RateLimitSettings;
use firnflow_core::StorageRoot;

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

fn config_with_cache_path(path: PathBuf) -> AppConfig {
    AppConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        storage_root: StorageRoot::s3_bucket("firnflow-offline").unwrap(),
        cache_memory_bytes: 1024 * 1024,
        cache_nvme_path: path,
        cache_nvme_bytes: 4 * 1024 * 1024,
        max_body_bytes: 16 * 1024 * 1024,
        storage_options: HashMap::new(),
        api_key: None,
        admin_api_key: None,
        metrics_token: None,
        rate_limit: RateLimitSettings::default(),
        object_cache_enabled: false,
        object_cache_dir: std::env::temp_dir(),
        object_cache_bytes: 0,
        object_cache_max_entry_bytes: 0,
        import_max_bytes: 0,
        import_tmp_dir: std::env::temp_dir(),
    }
}

/// The directory cannot be created at all. Hits the `create_dir_all`
/// guard, which already named the path and now names the variable too.
#[tokio::test]
async fn uncreatable_cache_dir_error_names_path_and_variable() {
    let blocker = std::env::temp_dir().join(format!("firnflow-blocker-{}", unique_suffix()));
    std::fs::write(&blocker, b"a regular file, not a directory").expect("write blocker");
    let unusable = blocker.join("cache");

    // `AppState` has no `Debug`, so match rather than `expect_err`.
    let rendered = match build_state(&config_with_cache_path(unusable.clone())).await {
        Ok(_) => {
            std::fs::remove_file(&blocker).ok();
            panic!("a cache path under a regular file must fail startup");
        }
        Err(e) => format!("{e:#}"),
    };

    std::fs::remove_file(&blocker).ok();

    assert!(
        rendered.contains(&unusable.display().to_string()),
        "error must name the directory it tried to use, got: {rendered}"
    );
    assert!(
        rendered.contains("FIRNFLOW_CACHE_NVME_PATH"),
        "error must name the variable that selected it, got: {rendered}"
    );
}

/// The directory is usable but building the cache inside it fails. This
/// is the path the reported failure took: the container image
/// pre-creates its default cache directory, so the directory check
/// passes and the first real failure comes from the cache itself, whose
/// error carried no path at all.
///
/// An oversized capacity stands in for the reporter's read-only
/// filesystem. Reproducing `EROFS` needs a genuinely read-only mount,
/// which a test cannot arrange portably: making the directory
/// permission-denied instead is not equivalent, because the cache does
/// not create its block file during startup and so does not notice.
/// What both cases share is the part this asserts, that a failure to
/// build the cache reports the directory and the variable that chose it
/// rather than a bare IO error.
#[tokio::test]
async fn cache_build_failure_error_names_path_and_variable() {
    let dir = std::env::temp_dir().join(format!("firnflow-cachefail-{}", unique_suffix()));
    std::fs::create_dir_all(&dir).expect("create cache dir");

    let mut cfg = config_with_cache_path(dir.clone());
    cfg.cache_nvme_bytes = usize::MAX;

    let result = build_state(&cfg).await;
    let rendered = match result {
        Ok(_) => {
            std::fs::remove_dir_all(&dir).ok();
            panic!("an unsatisfiable cache capacity must fail startup");
        }
        Err(e) => format!("{e:#}"),
    };
    std::fs::remove_dir_all(&dir).ok();

    assert!(
        rendered.contains(&dir.display().to_string()),
        "error must name the directory it tried to use, got: {rendered}"
    );
    assert!(
        rendered.contains("FIRNFLOW_CACHE_NVME_PATH"),
        "error must name the variable that selected it, got: {rendered}"
    );
}
