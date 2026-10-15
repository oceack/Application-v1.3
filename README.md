# Amsterdam Waste Logistics Optimizer v1.2

**Epsilon-Constraint Optimization**

This version uses the **ε-constraint method** instead of α-weighted multi-objective optimization.

## Key Differences from v1

| Aspect | v1 (Alpha-weighted) | v1.2 (ε-Constraint) |
|--------|---------------------|---------------------|
| **Objective** | Minimize `(1-α)·Cost + α·Penalty` | Minimize `Cost` |
| **Trade-off Control** | Weight parameter α ∈ [0,1] | Constraint bound ε |
| **User Input** | Abstract weight values | Interpretable penalty limits |
| **Pareto Frontier** | May miss non-convex regions | Can capture full frontier |

## Running the Application

```bash
cd "Application v1.2"
streamlit run frondend_optimization.py
```

## Epsilon Values

The ε (epsilon) parameter specifies the **maximum allowed total walking penalty**. Lower ε means:
- Stricter distance constraints
- Shorter walking distances for households
- Higher infrastructure cost (more facilities needed)

Suggested starting values: `1000, 5000, 10000, 50000, 100000, 500000`

## Mathematical Formulation

```
minimize    TotalCost = WaterFixedCosts + WaterVariableCosts + MobileFleetCosts

subject to: TotalWalkingPenalty ≤ ε
            + capacity constraints
            + assignment constraints
            + fleet logistics constraints
```

## Files

- `frondend_optimization.py` - Streamlit frontend
- `backend/` - Optimization backend
  - `model_builder.py` - Pyomo model with ε-constraint
  - `solver.py` - Optimization loop
  - `parameters.py` - Default parameters

## Requirements

Same as v1. See main project requirements.
