use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SuiteError {
    #[error("could not read suite: {0}")]
    Read(String),
    #[error("invalid suite manifest")]
    Manifest,
    #[error("invalid case at line {0}")]
    Case(usize),
    #[error("duplicate case id: {0}")]
    DuplicateId(String),
    #[error("suite contains no cases")]
    Empty,
}

#[derive(Debug, Clone, Deserialize)]
struct SuiteManifest {
    version: String,
    dataset: String,
    cases: String,
    scenarios: Vec<String>,
    scoring_policy: String,
}

#[derive(Debug, Clone)]
pub struct TestSuite {
    pub version: String,
    pub dataset: String,
    pub scoring_policy: String,
    pub cases: Vec<DecisionCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCase {
    pub case_id: String,
    pub scenario: String,
    pub input: Value,
    pub expected: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkUnit {
    pub model_id: String,
    pub case_id: String,
    pub repeat: u32,
}

impl WorkUnit {
    pub fn identity(&self) -> String {
        format!("{}/{}/{}", self.model_id, self.case_id, self.repeat)
    }
}

impl TestSuite {
    pub fn from_file(
        version: &str,
        dataset: &str,
        scoring_policy: &str,
        path: impl AsRef<Path>,
    ) -> Result<Self, SuiteError> {
        let jsonl = fs::read_to_string(path).map_err(|e| SuiteError::Read(e.to_string()))?;
        let mut suite = Self::from_jsonl(version, &jsonl)?;
        suite.dataset = dataset.to_owned();
        suite.scoring_policy = scoring_policy.to_owned();
        Ok(suite)
    }

    pub fn load(manifest_path: impl AsRef<Path>) -> Result<Self, SuiteError> {
        let path = manifest_path.as_ref();
        let text = fs::read_to_string(path).map_err(|e| SuiteError::Read(e.to_string()))?;
        let manifest: SuiteManifest = toml::from_str(&text).map_err(|_| SuiteError::Manifest)?;
        if manifest.version.trim().is_empty()
            || manifest.dataset.trim().is_empty()
            || manifest.scoring_policy.trim().is_empty()
            || manifest.scenarios.is_empty()
        {
            return Err(SuiteError::Manifest);
        }
        let case_path = path.parent().unwrap_or(Path::new(".")).join(manifest.cases);
        let jsonl = fs::read_to_string(case_path).map_err(|e| SuiteError::Read(e.to_string()))?;
        let suite = Self::from_jsonl(&manifest.version, &jsonl)?;
        if suite
            .cases
            .iter()
            .any(|case| !manifest.scenarios.contains(&case.scenario))
        {
            return Err(SuiteError::Manifest);
        }
        Ok(Self {
            dataset: manifest.dataset,
            scoring_policy: manifest.scoring_policy,
            ..suite
        })
    }

    pub fn from_jsonl(version: &str, jsonl: &str) -> Result<Self, SuiteError> {
        let mut cases = Vec::new();
        let mut ids = BTreeSet::new();
        for (index, line) in jsonl.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let case: DecisionCase =
                serde_json::from_str(line).map_err(|_| SuiteError::Case(index + 1))?;
            if case.case_id.trim().is_empty()
                || case.scenario.trim().is_empty()
                || !case.input.is_object()
                || !case.expected.is_object()
                || case.expected.as_object().is_some_and(|m| m.is_empty())
            {
                return Err(SuiteError::Case(index + 1));
            }
            if !ids.insert(case.case_id.clone()) {
                return Err(SuiteError::DuplicateId(case.case_id));
            }
            cases.push(case);
        }
        if cases.is_empty() {
            return Err(SuiteError::Empty);
        }
        Ok(Self {
            version: version.to_owned(),
            dataset: version.to_owned(),
            scoring_policy: "decision-v1".into(),
            cases,
        })
    }

    pub fn scenarios(&self) -> BTreeSet<String> {
        self.cases
            .iter()
            .map(|case| case.scenario.clone())
            .collect()
    }

    pub fn output_schema(&self, scenario: &str) -> Value {
        let mut fields: std::collections::BTreeMap<String, BTreeSet<String>> =
            std::collections::BTreeMap::new();
        for case in self.cases.iter().filter(|case| case.scenario == scenario) {
            if let Some(expected) = case.expected.as_object() {
                for (field, label) in expected {
                    if let Some(label) = label.as_str() {
                        fields
                            .entry(field.clone())
                            .or_default()
                            .insert(label.to_owned());
                    }
                }
            }
        }
        Value::Object(
            fields
                .into_iter()
                .map(|(field, values)| {
                    (
                        field,
                        Value::Array(values.into_iter().map(Value::String).collect()),
                    )
                })
                .collect(),
        )
    }

    pub fn work_units(&self, model_ids: &[String], repeats: u32) -> Vec<WorkUnit> {
        let mut units = Vec::new();
        for repeat in 1..=repeats {
            for case in &self.cases {
                for model_id in model_ids {
                    units.push(WorkUnit {
                        model_id: model_id.clone(),
                        case_id: case.case_id.clone(),
                        repeat,
                    });
                }
            }
        }
        units
    }
}
