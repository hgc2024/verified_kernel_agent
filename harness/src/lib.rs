//! `kharness`: compile -> verify -> profile harness for candidate CUDA kernels.
//!
//! The harness is the grader for the agent, so it is kept separate from the agent (different
//! language, different process) and treats every candidate as untrusted.

pub mod compile;
pub mod guards;
pub mod profile;
pub mod report;
pub mod sandbox;
pub mod task;
pub mod timing;
pub mod verify;

/// Errors that callers may want to match on, as opposed to plain `anyhow` context.
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// A stage that has not been built yet. Stubs return this instead of fake data.
    #[error("not implemented yet: {what} (TODO(phase-{phase}))")]
    NotImplemented { what: &'static str, phase: u8 },
}

/// Shorthand for stubs: an explicit error naming the phase that will implement `what`.
pub fn not_implemented<T>(what: &'static str, phase: u8) -> anyhow::Result<T> {
    Err(HarnessError::NotImplemented { what, phase }.into())
}
