# backend/parameters.py
"""
Parameter definitions, defaults, and validation for the waste logistics optimizer.

This module provides:
    - DEFAULT_PARAMS: Complete default parameter dictionary
    - validate_params(): Validation function with clear error messages
    - get_params(): Merge user params with defaults and validate

Usage:
    from backend.parameters import get_params, DEFAULT_PARAMS
    
    user_params = {'R_W': 300, 'R_M': 150}
    params = get_params(user_params)  # Merges with defaults, validates
"""

# needed as the parameters can be different types
from typing import Any

# DEFAULT PARAMETERS

DEFAULT_PARAMS: dict[str, Any] = {
    # --- Distance Constraints (meters) ---
    'R_W': 250,           # Max walking distance to water facility
    'R_W_ideal': 200,     # Ideal distance (penalty starts beyond this)
    'R_M': 100,           # Max walking distance to mobile facility
    'R_M_ideal': 50,      # Ideal distance for mobile (thesis value)
    'R_U': {
        'rest':      200,
        'cardboard': 200,
        'glass':     300,
    },
    'R_U_ideal': {
        'rest':      150,
        'cardboard': 150,
        'glass':     250,
    },
    
    # --- Cost Parameters (€) ---
    'f_W': 505.0,         # Fixed cost per open water location (€/week)
    'c_W_empty': 2000.0,    # Variable cost per boat emptying trip (€/trip)
    'C_shift': 450.0,     # Cost per fleet shift (€/shift)
    
    # --- Fleet Logistics ---
    'S_max': 8,           # Max stops per shift
    'F_freq': 2,          # Daily visits per mobile location
    'D_week': 6,          # Operating days per week
    'stop_cost_dump': 2.0,  # Stops lost per replenishment (dump) trip
    
    # --- Preference Bias ---
    'p_bias': 0.0,     # Penalty added to mobile to prefer water (higher = stronger water preference)

    # --- Per-fraction penalty weights ---
    # Multiplier applied to the walking-distance penalty per fraction. Default is 1.0
    # for every fraction (matches the thesis-style symmetric penalty). Setting a
    # fraction's weight to 0 turns off the walking-penalty for that fraction so that
    # only the hard R_max constraint remains; intermediate values can be used to
    # approximate frequency-weighted penalties (rest is disposed of more often than
    # cardboard, glass is the least frequent).
    'penalty_weight': {
        'rest':      1.0,
        'cardboard': 1.0,
        'glass':     1.0,
    },
    
    # --- Split Ratios (fraction going to water in split strategy) ---
    'beta': {
        'rest': 0.5,       # Residual waste: 50% to water, 50% to mobile
        'cardboard': 1.0,   # Cardboard: 100% to water (no split)
        'glass': 1.0,       # Glass: 100% to water (no split)
    },
    
    # --- Container Capacities (liters) ---
    'C_W': {
        'rest': 4500,       # Water container capacity for residual
        'cardboard': 250,   # Water container capacity for cardboard
        'glass': 1500,      # Water container capacity for glass
    },
    'C_M': {
        'rest': 1800,       # Mobile point capacity for residual
        'cardboard': 240,   # Mobile point capacity for cardboard
        'glass': 1200,      # Mobile point capacity for glass
    },
    
    # --- Truck Compartment Volumes (liters per trip) ---
    'V_truck': {
        'rest': 150,        # Truck volume for residual waste
        'cardboard': 20,    # Truck volume for cardboard
        'glass': 100,       # Truck volume for glass
    },

    # --- Underground Container parameters ---
    # Weekly costs include amortisation of purchase price + operational costs:
    #   regular: 60 EUR/week  (5 m^3)
    #   press:   190 EUR/week (12.5 m^3)
    # Press is only available for the rest and cardboard fractions; glass UCs are
    # always regular (enforced in model_builder via a press[u] == 0 constraint
    # for u in U_glass).
    # UCs cannot be emptied during the week -> weekly load <= physical UC capacity.
    'press_enabled': True,  # Global toggle: when False, model forces press[u]=0 for all UCs

    'f_U': {
        'rest':      60.0,
        'cardboard': 60.0,
        'glass':     60.0,
    },
    'f_U_press': {
        'rest':      190.0,
        'cardboard': 190.0,
        'glass':     60.0,   # glass has no press; same as regular (press forced to 0 in model)
    },
    # UC capacities in kg/week — physical volume (5 m^3) x bulk density:
    #   rest      5 m^3 * 90 kg/m^3 = 450 kg
    #   cardboard 5 m^3 * 25 kg/m^3 = 125 kg
    #   glass     5 m^3 * 275 kg/m^3 = 1375 kg
    # Press 2.5x volume = 12.5 m^3 (only rest + cardboard):
    'C_U': {
        'rest':      450,
        'cardboard': 125,
        'glass':     1375,
    },
    'C_U_press': {
        'rest':      1125,    # 12.5 m^3 * 90 kg/m^3
        'cardboard': 312.5,   # 12.5 m^3 * 25 kg/m^3
        'glass':     1375,    # unused (press disabled for glass); kept equal to regular
    },

    # Waste densities (kg / m^3) — informational, kept for downstream conversions
    # if needed (model itself works in kg, not in volume).
    'density_kg_m3': {
        'rest':      90,
        'cardboard': 25,
        'glass':     275,
    },
}

# AUGMECON2 Algorithm Defaults
AUGMECON_DEFAULTS: dict[str, Any] = {
    'n_grid_points': 10,      # Number of Pareto points to generate
    'delta': 1e-3,            # Augmentation coefficient (ensures proper Pareto points)
    'early_exit': True,       # Stop when consecutive infeasibles detected
    'bypass_threshold': 3,    # Number of consecutive skips before early exit
}

# Waste types for iteration
WASTE_TYPES = ['rest', 'cardboard', 'glass']


# VALIDATION > checks the user input for wrong values

class ParameterError(ValueError):
    """Raised when parameter validation fails."""
    pass


def validate_params(params: dict[str, Any]) -> list[str]:
    """
    Validate parameter dictionary.
    
    Args:
        params: Parameter dictionary to validate
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # --- Distance Constraints ---
    if params.get('R_W', 0) <= 0:
        errors.append("R_W (max water radius) must be positive")
    if params.get('R_M', 0) <= 0:
        errors.append("R_M (max mobile radius) must be positive")
    
    if params.get('R_W_ideal', 0) > params.get('R_W', 0):
        errors.append(f"R_W_ideal ({params.get('R_W_ideal')}) cannot exceed R_W ({params.get('R_W')})")
    if params.get('R_M_ideal', 0) > params.get('R_M', 0):
        errors.append(f"R_M_ideal ({params.get('R_M_ideal')}) cannot exceed R_M ({params.get('R_M')})")
    R_U = params.get('R_U', {})
    R_U_ideal = params.get('R_U_ideal', {})
    for waste_type in WASTE_TYPES:
        ru = R_U.get(waste_type, 0) if isinstance(R_U, dict) else R_U
        ru_ideal = R_U_ideal.get(waste_type, 0) if isinstance(R_U_ideal, dict) else R_U_ideal
        if ru <= 0:
            errors.append(f"R_U['{waste_type}'] (max UC radius) must be positive")
        if ru_ideal > ru:
            errors.append(f"R_U_ideal['{waste_type}'] ({ru_ideal}) cannot exceed R_U['{waste_type}'] ({ru})")
    
    # --- Cost Parameters ---
    if params.get('f_W', 0) < 0:
        errors.append("f_W (fixed water cost) cannot be negative")
    if params.get('c_W_empty', 0) < 0:
        errors.append("c_W_empty (emptying cost) cannot be negative")
    if params.get('C_shift', 0) < 0:
        errors.append("C_shift (shift cost) cannot be negative")
    
    # --- Fleet Logistics ---
    if params.get('S_max', 0) <= 0:
        errors.append("S_max (stops per shift) must be positive")
    if params.get('F_freq', 0) <= 0:
        errors.append("F_freq (visits per day) must be positive")
    if params.get('D_week', 0) <= 0 or params.get('D_week', 0) > 7:
        errors.append("D_week (days per week) must be between 1 and 7")
    if params.get('stop_cost_dump', 0) < 0:
        errors.append("stop_cost_dump cannot be negative")
    
    # --- Beta (Split Ratios) ---
    beta = params.get('beta', {})
    for waste_type in WASTE_TYPES:
        b = beta.get(waste_type, 0)
        if b < 0 or b > 1:
            errors.append(f"beta['{waste_type}'] must be between 0 and 1 (got {b})")
    
    # --- Capacities ---
    for cap_name in ['C_W', 'C_M', 'V_truck', 'C_U', 'C_U_press']:
        cap_dict = params.get(cap_name, {})
        for waste_type in WASTE_TYPES:
            val = cap_dict.get(waste_type, 0)
            if val <= 0:
                errors.append(f"{cap_name}['{waste_type}'] must be positive (got {val})")

    # Press capacity must be >= regular (otherwise press is strictly worse)
    C_U = params.get('C_U', {})
    C_U_press = params.get('C_U_press', {})
    for waste_type in WASTE_TYPES:
        if C_U.get(waste_type, 0) > C_U_press.get(waste_type, 0):
            errors.append(
                f"C_U_press['{waste_type}'] ({C_U_press.get(waste_type)}) "
                f"must be >= C_U['{waste_type}'] ({C_U.get(waste_type)}); "
                f"press containers can't have smaller capacity than regular"
            )

    # --- UC fixed costs ---
    for cost_name in ['f_U', 'f_U_press']:
        cost_dict = params.get(cost_name, {})
        for waste_type in WASTE_TYPES:
            val = cost_dict.get(waste_type, 0)
            if val < 0:
                errors.append(f"{cost_name}['{waste_type}'] cannot be negative (got {val})")
    
    # --- Bias ---
    if params.get('p_bias', 0) < 0:
        errors.append("p_bias cannot be negative")

    # --- Penalty weight per fraction (>= 0) ---
    pw = params.get('penalty_weight', {})
    for waste_type in WASTE_TYPES:
        v = pw.get(waste_type, 1.0)
        if v < 0:
            errors.append(f"penalty_weight['{waste_type}'] must be non-negative (got {v})")

    return errors


def get_params(user_params: dict[str, Any] | None = None, raise_on_error: bool = True) -> dict[str, Any]:
    """
    Merge user parameters with defaults and validate.
    So when you input a different value in the gui, this function will override the default values with the user values.
    
    Args:
        user_params: User-provided parameter overrides (can be partial)
        raise_on_error: If True, raise ParameterError on validation failure
        
    Returns:
        Complete, validated parameter dictionary
        
    Raises:
        ParameterError: If validation fails and raise_on_error is True
    """
    # Start with defaults
    params = deep_copy_params(DEFAULT_PARAMS)
    
    # Merge user params
    if user_params:
        for key, value in user_params.items():
            if key in params:
                # Handle nested dicts (beta, C_W, C_M, V_truck)
                if isinstance(params[key], dict) and isinstance(value, dict):
                    params[key].update(value)
                else:
                    params[key] = value
            else:
                # Unknown parameter — add it anyway (flexibility for extensions)
                params[key] = value
    
    # Validate
    errors = validate_params(params)
    if errors and raise_on_error:
        raise ParameterError("Parameter validation failed:\n  - " + "\n  - ".join(errors))
    
    return params


def deep_copy_params(params: dict[str, Any]) -> dict[str, Any]:
    """Create a deep copy of the parameter dictionary."""
    result = {}
    for key, value in params.items():
        if isinstance(value, dict):
            result[key] = dict(value)
        else:
            result[key] = value
    return result


# ==========================================
# EPSILON VALIDATION (for epsilon-constraint method)
# ==========================================

def validate_epsilons(epsilons: list[float]) -> list[str]:
    """
    Validate epsilon values for epsilon-constraint optimization.
    
    Args:
        epsilons: List of epsilon values (penalty constraint bounds)
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    if not epsilons:
        errors.append("At least one epsilon value is required")
        return errors
    
    for i, eps in enumerate(epsilons):
        if not isinstance(eps, (int, float)):
            errors.append(f"Epsilon[{i}] must be a number (got {type(eps).__name__})")
        elif eps < 0:
            errors.append(f"Epsilon[{i}] = {eps} must be non-negative")
    
    return errors


def parse_epsilon_string(epsilon_str: str) -> list[float]:
    """
    Parse comma-separated epsilon string from UI.
    
    Args:
        epsilon_str: String like "5000, 10000, 20000, 50000"
        
    Returns:
        List of floats
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        epsilons = [float(x.strip()) for x in epsilon_str.split(',') if x.strip()]
        return epsilons
    except ValueError as e:
        raise ValueError(f"Could not parse epsilon values: {e}")
