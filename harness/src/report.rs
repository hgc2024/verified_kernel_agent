//! JSON result schema (README: Harness > Result schema).
//! One `RunReport` is printed per `kharness` invocation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    pub task_id: String,
    /// Candidate path or content hash.
    pub candidate: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile: Option<CompileReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify: Option<VerifyReport>,
    #[serde(default)]
    pub guards: Vec<GuardResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<ProfileReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<EnvReport>,
    /// OS-derived seed used for input generation; recorded, never shown to the candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl RunReport {
    pub fn new(task_id: &str, candidate: &str) -> Self {
        Self {
            task_id: task_id.to_owned(),
            candidate: candidate.to_owned(),
            compile: None,
            verify: None,
            guards: Vec::new(),
            timing: None,
            profile: None,
            env: None,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompileReport {
    pub ok: bool,
    pub stderr: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub trials: u32,
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    pub failures: Vec<VerifyFailure>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyFailure {
    pub shape: BTreeMap<String, i64>,
    pub seed: u64,
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    /// Flat index of the first element outside tolerance, if any.
    pub first_mismatch_index: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardResult {
    pub name: String,
    pub ok: bool,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingReport {
    pub median_ms: f64,
    pub p10_ms: f64,
    pub p90_ms: f64,
    pub ci95_ms: [f64; 2],
    pub baseline_median_ms: Option<f64>,
    pub speedup: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileReport {
    pub metrics: BTreeMap<String, f64>,
}

/// Hardware and software environment, always detected at runtime and never hardcoded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvReport {
    pub gpu: String,
    pub driver: String,
    pub cuda: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_round_trips() {
        let report = RunReport::new("level1.gemm_fp32", "cand.cu");
        let json = serde_json::to_string(&report).unwrap();
        let back: RunReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back);
    }

    #[test]
    fn full_report_round_trips() {
        // Synthetic values exercising the schema only; not measurements.
        let mut report = RunReport::new("level1.gemm_fp32", "cand.cu");
        report.compile = Some(CompileReport {
            ok: true,
            stderr: String::new(),
        });
        report.verify = Some(VerifyReport {
            ok: false,
            trials: 2,
            max_abs_err: 0.5,
            max_rel_err: 0.25,
            failures: vec![VerifyFailure {
                shape: BTreeMap::from([("M".to_owned(), 3), ("N".to_owned(), 5)]),
                seed: 7,
                max_abs_err: 0.5,
                max_rel_err: 0.25,
                first_mismatch_index: Some(4),
            }],
        });
        report.guards = vec![GuardResult {
            name: "output_poisoning".to_owned(),
            ok: true,
            evidence: "synthetic".to_owned(),
        }];
        report.timing = Some(TimingReport {
            median_ms: 1.0,
            p10_ms: 0.5,
            p90_ms: 1.5,
            ci95_ms: [0.75, 1.25],
            baseline_median_ms: Some(2.0),
            speedup: Some(2.0),
        });
        report.profile = Some(ProfileReport {
            metrics: BTreeMap::from([("metric".to_owned(), 1.0)]),
        });
        report.env = Some(EnvReport {
            gpu: "synthetic".to_owned(),
            driver: "synthetic".to_owned(),
            cuda: "synthetic".to_owned(),
        });
        report.seed = Some(123);

        let json = serde_json::to_string(&report).unwrap();
        let back: RunReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back);
    }

    #[test]
    fn absent_sections_are_omitted() {
        let json = serde_json::to_value(RunReport::new("t", "c")).unwrap();
        let obj = json.as_object().unwrap();
        assert!(!obj.contains_key("compile"));
        assert!(!obj.contains_key("timing"));
        assert!(obj.contains_key("guards"));
    }
}
