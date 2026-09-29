//! Subprocess isolation for nvcc and the runner: wall-clock timeout, scrubbed environment,
//! per-run temp directories, and best-effort memory and network limits.

use std::process::{Command, Output};
use std::time::Duration;

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub timeout: Duration,
    /// Address-space cap for the child process, if the platform supports one.
    pub memory_limit_bytes: Option<u64>,
    /// Remove all environment variables except an explicit allow-list (PATH, CUDA_*).
    pub scrub_env: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            memory_limit_bytes: None,
            scrub_env: true,
        }
    }
}

pub fn run(_command: &mut Command, _config: &SandboxConfig) -> Result<Output> {
    // TODO(phase-2): spawn with a scrubbed env and temp cwd, kill on timeout.
    crate::not_implemented("sandbox::run", 2)
}
