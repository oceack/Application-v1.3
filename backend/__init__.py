# backend/__init__.py
"""
Backend package for the Amsterdam Waste Logistics Optimizer (v1.2 - Epsilon-Constraint + AUGMECON2).
This file makes sure that each file does not have to be imported separately.

This package shows the frontend the following functions:
    - run_optimization - standard epsilon-constraint method
    - run_augmecon - AUGMECON2 method with payoff table and acceleration
    - compute_payoff_table - compute objective ranges
    - run_diagnostics - diagnose infeasibility causes when optimization fails
    - DEFAULT_PARAMS, AUGMECON_DEFAULTS, get_params, validate_params

Files covered by this package:
    - data_loader: CSV loading, coordinate extraction, ID cleaning
    - model_builder: Pyomo MILP model construction
    - solver: Optimization loop and result extraction
    - diagnostics: Feasibility checks with slack-variable relaxation
    - parameters: Default values, validation, type definitions
"""

from .solver import run_optimization, run_augmecon, compute_payoff_table
from .diagnostics import run_diagnostics
from .parameters import (
    DEFAULT_PARAMS,
    AUGMECON_DEFAULTS,
    get_params,
    validate_params,
    validate_epsilons,
    parse_epsilon_string,
    ParameterError,
)

# If using import * from backend, this will all be imported > Recommend you not to do this
__all__ = [
    "run_optimization",
    "run_augmecon",
    "compute_payoff_table",
    "run_diagnostics",
    "DEFAULT_PARAMS",
    "AUGMECON_DEFAULTS",
    "get_params",
    "validate_params",
    "validate_epsilons",
    "parse_epsilon_string",
    "ParameterError",
]

