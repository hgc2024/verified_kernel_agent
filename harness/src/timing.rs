//! Parses runner timings and summarizes them: median, p10/p90, 95% bootstrap CI, and speedup
//! over the task baseline. Methodology: docs/timing_methodology.md.

use std::path::Path;

use anyhow::Result;

use crate::report::TimingReport;
use crate::task::Task;

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub warmup: u32,
    pub iters: u32,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            warmup: 10,
            iters: 100,
        }
    }
}

pub fn bench(_task: &Task, _candidate: &Path, _opts: &BenchOptions) -> Result<TimingReport> {
    // TODO(phase-2): run candidate and baseline through the runner and summarize the samples.
    crate::not_implemented("bench", 2)
}
