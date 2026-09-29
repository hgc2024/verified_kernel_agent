//! nvcc invocation for candidates: timeouts and captured diagnostics.
//!
//! A compile error is a normal outcome (`CompileReport { ok: false, .. }`) that the agent learns
//! from; only harness-side failures (nvcc missing, sandbox error) are `Err`.

use std::path::Path;

use anyhow::Result;

use crate::report::CompileReport;
use crate::task::Task;

pub fn compile(_task: &Task, _candidate: &Path) -> Result<CompileReport> {
    // TODO(phase-2): run nvcc through `sandbox::run` into a temp dir as a shared library.
    crate::not_implemented("compile", 2)
}
