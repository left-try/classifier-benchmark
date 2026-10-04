use evolving_benchmark::domain::suite::TestSuite;

#[test]
fn bundled_decision_suite_has_30_cases_and_six_scenarios() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    assert_eq!(suite.cases.len(), 30);
    assert_eq!(suite.scenarios().len(), 6);
    assert_eq!(suite.version, "v1");
}

#[test]
fn rejects_duplicate_case_ids() {
    let result = TestSuite::from_jsonl(
        "v-test",
        "{\"case_id\":\"same\",\"scenario\":\"x\",\"input\":{},\"expected\":{\"v\":1}}\n{\"case_id\":\"same\",\"scenario\":\"x\",\"input\":{},\"expected\":{\"v\":1}}",
    );
    assert!(result.is_err());
}

#[test]
fn work_units_have_stable_model_case_repeat_identity() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let units = suite.work_units(&["provider/model-a".into()], 2);
    assert_eq!(units.len(), 60);
    assert_eq!(units[0].identity(), "provider/model-a/TS-001/1");
}
