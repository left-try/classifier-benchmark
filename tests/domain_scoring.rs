use evolving_benchmark::domain::scoring::ScorePolicy;

#[test]
fn decision_v1_matches_documented_cost_and_latency_factors() {
    let score = ScorePolicy::decision_v1().score(1.0, 0.001, 500.0).unwrap();
    assert!((score.cost_factor - 0.5).abs() < 1e-12);
    assert_eq!(score.latency_factor, 1.0);
    assert!((score.score - 0.5).abs() < 1e-12);
}

#[test]
fn decision_v1_rejects_negative_cost_and_out_of_range_accuracy() {
    let policy = ScorePolicy::decision_v1();
    assert!(policy.score(1.0, -0.1, 100.0).is_err());
    assert!(policy.score(1.1, 0.1, 100.0).is_err());
}
