use crate::{
    application::plan::RunPlan,
    domain::{model::ModelCandidate, suite::WorkUnit},
};
use std::collections::HashMap;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Money(pub u64); // integer micro-USD

impl Money {
    pub fn from_usd_ceil(value: f64) -> Result<Self, BudgetError> {
        if !value.is_finite() || value < 0.0 || value * 1_000_000.0 > u64::MAX as f64 {
            return Err(BudgetError::InvalidAmount);
        }
        Ok(Self((value * 1_000_000.0).ceil() as u64))
    }
    pub fn as_usd(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
    pub fn checked_add(self, other: Self) -> Result<Self, BudgetError> {
        self.0
            .checked_add(other.0)
            .map(Self)
            .ok_or(BudgetError::InvalidAmount)
    }
    pub fn checked_mul(self, count: u64) -> Result<Self, BudgetError> {
        self.0
            .checked_mul(count)
            .map(Self)
            .ok_or(BudgetError::InvalidAmount)
    }
}

#[derive(Debug, Error)]
pub enum EstimateError {
    #[error("catalog pricing is unavailable")]
    MissingPrice,
    #[error("cost estimate is invalid or overflowed")]
    Invalid,
}

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("budget amount is invalid")]
    InvalidAmount,
    #[error("reservation would exceed remaining budget")]
    Exceeded,
    #[error("reservation was not found")]
    ReservationNotFound,
    #[error("budget ledger lock is poisoned")]
    Poisoned,
}

pub struct CostEstimator;

impl CostEstimator {
    pub fn estimate(
        model: &ModelCandidate,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<Money, EstimateError> {
        let input = model
            .prompt_usd_per_million
            .ok_or(EstimateError::MissingPrice)?;
        let output = model
            .completion_usd_per_million
            .ok_or(EstimateError::MissingPrice)?;
        if !input.is_finite() || input < 0.0 || !output.is_finite() || output < 0.0 {
            return Err(EstimateError::Invalid);
        }
        // USD/token price × token count, converted to micro-USD (price is normalized per million).
        let micros = input * input_tokens as f64 + output * output_tokens as f64;
        if !micros.is_finite() || micros > u64::MAX as f64 {
            return Err(EstimateError::Invalid);
        }
        Ok(Money(micros.ceil() as u64))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReservationId(pub u64);

#[derive(Default)]
struct LedgerState {
    reserved: Money,
    spent: Money,
    next_id: u64,
    reservations: HashMap<ReservationId, Money>,
    unknown_cost: bool,
}

pub struct BudgetLedger {
    budget: Money,
    state: Mutex<LedgerState>,
}

impl BudgetLedger {
    pub fn new(budget: Money) -> Self {
        Self {
            budget,
            state: Mutex::new(LedgerState::default()),
        }
    }

    pub fn with_existing(budget: Money, spent: Money, unknown_cost: bool) -> Self {
        let state = LedgerState {
            spent,
            unknown_cost,
            ..LedgerState::default()
        };
        Self {
            budget,
            state: Mutex::new(state),
        }
    }

    pub fn reserve(&self, _work_key: &str, amount: Money) -> Result<ReservationId, BudgetError> {
        let mut state = self.state.lock().map_err(|_| BudgetError::Poisoned)?;
        let committed = state.spent.checked_add(state.reserved)?;
        if committed.checked_add(amount)? > self.budget {
            return Err(BudgetError::Exceeded);
        }
        state.next_id += 1;
        let id = ReservationId(state.next_id);
        state.reserved = state.reserved.checked_add(amount)?;
        state.reservations.insert(id, amount);
        Ok(id)
    }

    pub fn reconcile(&self, id: ReservationId, actual: Option<Money>) -> Result<(), BudgetError> {
        let mut state = self.state.lock().map_err(|_| BudgetError::Poisoned)?;
        let reserved = state
            .reservations
            .remove(&id)
            .ok_or(BudgetError::ReservationNotFound)?;
        state.reserved = Money(state.reserved.0.saturating_sub(reserved.0));
        match actual {
            Some(value) => state.spent = state.spent.checked_add(value)?,
            None => state.unknown_cost = true,
        }
        Ok(())
    }

    pub fn spent(&self) -> Money {
        self.state
            .lock()
            .map(|s| s.spent)
            .unwrap_or(Money(u64::MAX))
    }
    pub fn reserved(&self) -> Money {
        self.state
            .lock()
            .map(|s| s.reserved)
            .unwrap_or(Money(u64::MAX))
    }
    pub fn may_schedule(&self) -> bool {
        self.state
            .lock()
            .map(|s| !s.unknown_cost && s.spent.0.saturating_add(s.reserved.0) < self.budget.0)
            .unwrap_or(false)
    }
    pub fn over_budget(&self) -> bool {
        self.state
            .lock()
            .map(|s| s.spent.0 > self.budget.0)
            .unwrap_or(true)
    }
    pub fn budget(&self) -> Money {
        self.budget
    }
}

pub fn estimate_plan(plan: &RunPlan) -> Result<Money, EstimateError> {
    let mut total = Money(0);
    for unit in &plan.work_units {
        let model = plan
            .models
            .iter()
            .find(|m| m.id == unit.model_id)
            .ok_or(EstimateError::Invalid)?;
        let input_bound = *plan
            .input_token_bounds
            .get(&unit.case_id)
            .ok_or(EstimateError::Invalid)?;
        let estimate = CostEstimator::estimate(model, input_bound, plan.max_output_tokens as u64)?;
        total = total
            .checked_add(
                estimate
                    .checked_mul(plan.max_retries as u64 + 1)
                    .map_err(|_| EstimateError::Invalid)?,
            )
            .map_err(|_| EstimateError::Invalid)?;
    }
    Ok(total)
}

pub fn reservation_key(unit: &WorkUnit) -> String {
    unit.identity()
}
