//! Anti-reward-hacking guards (README: Harness > Anti-reward-hacking guards).
//! Each produces a named pass/fail with evidence.
//! Exploits and their test fixtures: docs/anti_reward_hacking.md.

use std::path::Path;

use anyhow::Result;

use crate::report::GuardResult;
use crate::task::Task;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guard {
    /// Outputs are NaN-filled before each launch, so untouched outputs fail.
    OutputPoisoning,
    /// Identical outputs for different inputs indicate caching.
    InputVariation,
    /// Fuzzed shapes defeat special-casing of the nominal shape.
    ShapeVariation,
    /// Results are re-validated after a full device sync.
    Synchronization,
    /// Rejects forbidden libraries, file/env/process access, and references to the reference.
    StaticLint,
    /// Wall-clock timeout, memory cap, no network.
    ResourceLimits,
    /// Informational: whether outputs are bitwise identical across runs. Never a failure.
    Determinism,
}

impl Guard {
    pub const ALL: [Guard; 7] = [
        Guard::OutputPoisoning,
        Guard::InputVariation,
        Guard::ShapeVariation,
        Guard::Synchronization,
        Guard::StaticLint,
        Guard::ResourceLimits,
        Guard::Determinism,
    ];

    /// Stable name used in the JSON report.
    pub fn name(self) -> &'static str {
        match self {
            Guard::OutputPoisoning => "output_poisoning",
            Guard::InputVariation => "input_variation",
            Guard::ShapeVariation => "shape_variation",
            Guard::Synchronization => "synchronization",
            Guard::StaticLint => "static_lint",
            Guard::ResourceLimits => "resource_limits",
            Guard::Determinism => "determinism",
        }
    }

    /// Whether a failing result disqualifies the candidate.
    pub fn is_blocking(self) -> bool {
        !matches!(self, Guard::Determinism)
    }
}

pub fn run_all(_task: &Task, _candidate: &Path) -> Result<Vec<GuardResult>> {
    // TODO(phase-3): implement each guard and its cheating fixture in tests/fixtures/cheaters/.
    crate::not_implemented("guards", 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    #[test]
    fn guard_names_are_unique() {
        let names: HashSet<_> = Guard::ALL.iter().map(|g| g.name()).collect();
        assert_eq!(names.len(), Guard::ALL.len());
    }

    #[test]
    fn only_determinism_is_informational() {
        let informational: Vec<_> = Guard::ALL
            .into_iter()
            .filter(|g| !g.is_blocking())
            .collect();
        assert_eq!(informational, vec![Guard::Determinism]);
    }
}
