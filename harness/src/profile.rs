//! Optional Nsight Compute (`ncu`) invocation and metric extraction, fed back to the reviewer.

use std::path::Path;

use anyhow::Result;

use crate::report::ProfileReport;
use crate::task::Task;

#[derive(Debug, Clone)]
pub struct ProfileOptions {
    /// Named metric set (`default`) or a comma-separated list of ncu metric names.
    pub metrics: String,
}

pub fn profile(_task: &Task, _candidate: &Path, _opts: &ProfileOptions) -> Result<ProfileReport> {
    // TODO(phase-2): run the runner under `ncu --csv` and parse the requested metrics.
    crate::not_implemented("profile", 2)
}
