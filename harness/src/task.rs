//! Task TOML schema (README: Harness > Task schema) and loader.
//!
//! Validation happens at load time so a malformed task fails loudly before any candidate is
//! compiled, rather than surfacing later as a confusing verification failure.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    /// Shape symbols, e.g. `["M", "K"]`. Concrete sizes come from [`Shapes`].
    pub shape: Vec<String>,
}

/// Fuzzing ranges per shape symbol (`M = [lo, hi]`, inclusive) plus tile sizes that fuzzed
/// shapes must deliberately *not* be multiples of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzSpec {
    #[serde(default)]
    pub include_non_multiples_of: Vec<i64>,
    #[serde(flatten)]
    pub ranges: BTreeMap<String, [i64; 2]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shapes {
    pub nominal: BTreeMap<String, i64>,
    pub fuzz: FuzzSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    pub rtol: f64,
    pub atol: f64,
}

/// Task-level restrictions enforced by the static source lint guard (README: guard 5).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constraints {
    /// Libraries the candidate may not call, e.g. `["cublas", "cudnn", "thrust"]`.
    #[serde(default)]
    pub forbid: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub id: String,
    pub description: String,
    /// `module:function` of the reference implementation, e.g. `vka.reference:gemm`.
    pub reference: String,
    /// Path (relative to the repo root) of the baseline kernel that speedup is measured against.
    pub baseline: String,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub shapes: Shapes,
    pub tolerance: Tolerance,
    #[serde(default)]
    pub constraints: Constraints,
}

impl Task {
    pub fn from_toml_str(source: &str) -> Result<Self> {
        let task: Task = toml::from_str(source).context("parsing task TOML")?;
        task.validate()?;
        Ok(task)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("reading task file {}", path.display()))?;
        Self::from_toml_str(&source)
            .with_context(|| format!("loading task {}", path.display()))
    }

    /// Shape symbols in order of first appearance across inputs then outputs. This is the order
    /// of the `shape` array passed to `vka_launch` (see kernels/include/vka/kernels.h).
    pub fn shape_symbols(&self) -> Vec<&str> {
        let mut symbols: Vec<&str> = Vec::new();
        for tensor in self.inputs.iter().chain(&self.outputs) {
            for symbol in &tensor.shape {
                if !symbols.contains(&symbol.as_str()) {
                    symbols.push(symbol);
                }
            }
        }
        symbols
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            bail!("task id is empty");
        }
        if !self.reference.contains(':') {
            bail!(
                "reference {:?} must have the form `module:function`",
                self.reference
            );
        }
        if self.outputs.is_empty() {
            bail!("task {} declares no outputs", self.id);
        }
        for symbol in self.shape_symbols() {
            match self.shapes.nominal.get(symbol) {
                None => bail!("shape symbol {symbol:?} has no nominal size"),
                Some(size) if *size < 1 => {
                    bail!("nominal size of {symbol:?} must be >= 1, got {size}")
                }
                Some(_) => {}
            }
            match self.shapes.fuzz.ranges.get(symbol) {
                None => bail!("shape symbol {symbol:?} has no fuzz range"),
                Some([lo, hi]) if *lo < 1 || lo > hi => {
                    bail!("fuzz range of {symbol:?} needs 1 <= lo <= hi, got [{lo}, {hi}]")
                }
                Some(_) => {}
            }
        }
        let tol = &self.tolerance;
        if !(tol.rtol >= 0.0 && tol.atol >= 0.0) {
            bail!(
                "tolerances must be non-negative, got rtol={} atol={}",
                tol.rtol,
                tol.atol
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    const GEMM: &str = r#"
id = "level1.gemm_fp32"
description = "C = A @ B, row-major, FP32"
reference = "vka.reference:gemm"
baseline = "kernels/gemm_tiled.cu"
inputs  = [{ name = "A", dtype = "f32", shape = ["M","K"] },
           { name = "B", dtype = "f32", shape = ["K","N"] }]
outputs = [{ name = "C", dtype = "f32", shape = ["M","N"] }]
[shapes]
nominal = { M = 1024, N = 1024, K = 1024 }
fuzz    = { M = [1, 4096], N = [1, 4096], K = [1, 2048], include_non_multiples_of = [16, 32, 128] }
[tolerance]
rtol = 1e-4
atol = 1e-5
"#;

    #[test]
    fn parses_spec_example() {
        let task = Task::from_toml_str(GEMM).unwrap();
        assert_eq!(task.id, "level1.gemm_fp32");
        assert_eq!(task.inputs.len(), 2);
        assert_eq!(task.inputs[0].dtype, DType::F32);
        assert_eq!(task.shapes.nominal["K"], 1024);
        assert_eq!(task.shapes.fuzz.ranges["K"], [1, 2048]);
        assert_eq!(task.shapes.fuzz.include_non_multiples_of, vec![16, 32, 128]);
        assert!(!task.shapes.fuzz.ranges.contains_key("include_non_multiples_of"));
        assert!(task.constraints.forbid.is_empty());
    }

    #[test]
    fn shape_symbols_follow_first_appearance() {
        let task = Task::from_toml_str(GEMM).unwrap();
        assert_eq!(task.shape_symbols(), vec!["M", "K", "N"]);
    }

    #[test]
    fn rejects_symbol_without_nominal_size() {
        let source = GEMM.replace(
            "nominal = { M = 1024, N = 1024, K = 1024 }",
            "nominal = { M = 1024, N = 1024 }",
        );
        let err = Task::from_toml_str(&source).unwrap_err();
        assert!(format!("{err:#}").contains("\"K\""), "{err:#}");
    }

    #[test]
    fn rejects_inverted_fuzz_range() {
        let source = GEMM.replace("K = [1, 2048]", "K = [2048, 1]");
        assert!(Task::from_toml_str(&source).is_err());
    }

    #[test]
    fn rejects_unknown_fields() {
        let source = format!("surprise = true\n{GEMM}");
        assert!(Task::from_toml_str(&source).is_err());
    }

    fn collect_tomls(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect_tomls(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "toml") {
                out.push(path);
            }
        }
    }

    #[test]
    fn all_repo_tasks_load() {
        let tasks_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tasks");
        let mut files = Vec::new();
        collect_tomls(&tasks_dir, &mut files);
        assert!(!files.is_empty(), "no task files under {}", tasks_dir.display());
        for file in files {
            Task::load(&file).unwrap_or_else(|err| panic!("{err:#}"));
        }
    }
}
