//! Multi-trial correctness: the nominal shape plus fuzzed shapes (non-multiples of common tile
//! sizes, 1x1, tall-skinny), each with fresh OS-seeded random inputs.

use std::path::Path;

use anyhow::Result;

use crate::report::VerifyReport;
use crate::task::Task;

#[derive(Debug, Clone)]
pub struct VerifyOptions {
    pub trials: u32,
    pub seed_from_os: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            trials: 20,
            seed_from_os: true,
        }
    }
}

pub fn verify(_task: &Task, _candidate: &Path, _opts: &VerifyOptions) -> Result<VerifyReport> {
    // TODO(phase-2): generate trial shapes, drive the runner, compare against the reference.
    crate::not_implemented("verify", 2)
}
