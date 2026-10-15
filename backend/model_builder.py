# backend/model_builder.py
"""
Pyomo MILP model construction for the Tri-Modal Amsterdam Centrum waste
location-allocation problem.

Per-fraction mode: each (household, fraction) pair is independently assigned to
W, M, or U_t. A household may drop off cardboard at a water container and rest
at a mobile point — whatever serves each fraction best. No "primary mode" per
household, no W<->M split mechanism (the per-fraction structure handles it).

LP-relaxation of assignment vars: y_w, y_m, y_u, press, z_w are integer/binary;
x_w[i,w,t], x_m[i,m,t], x_u[i,t,u] are continuous in [0,1]. Aggregate facility
decisions and total cost/penalty are unaffected; individual fractional household
assignments can be rounded to argmax in post-processing.

Underground container decisions:
    y_u[u]   = 1 if UC u is activated (assignable to)
    press[u] = 1 if UC u is configured as a press container (higher capacity, higher cost)
A `params['press_enabled']` flag forces press[u] = 0 for all u when False.

Epsilon-Constraint vs AUGMECON2 (unchanged from thesis):
    standard:  min Cost                          s.t.  TotalPenalty <= epsilon
    AUGMECON2: min Cost - delta*(slack / range)  s.t.  TotalPenalty + slack == epsilon

Exports:
    build_fleet_model(...)
"""

import pyomo.environ as pyo

from .parameters import WASTE_TYPES


def build_fleet_model(
    I, W, M, U_t,
    feas_w, feas_m, feas_u_t,
    demand, params,
    pen_w, pen_m, pen_m_raw, pen_u_t,
    epsilon=None,
    dist_water=None, dist_mobile=None,
    split_enabled=False,            # kept in signature for caller compatibility (no-op now)
    augmecon_mode=False,
    penalty_range=None,
    delta=1e-3,
):
    """
    Args:
        I: list of household IDs.
        W, M: lists of water / mobile facility IDs (already prefixed 'W-', 'M-').
        U_t: dict {fraction -> list of UC IDs} (already prefixed 'U_<frac>-').
        feas_w, feas_m: list of feasible (i, w) / (i, m) pairs after R cutoff.
        feas_u_t: dict {fraction -> list of (i, u) pairs} after R_U cutoff.
        demand: dict {(i, fraction) -> volume in litres/week}.
        params: parameter dict (see backend.parameters.DEFAULT_PARAMS).
        pen_w, pen_m, pen_m_raw, pen_u_t: penalty dicts.
        epsilon: optional bound on total walking penalty.
        dist_water, dist_mobile: distance dicts (kept for caller compatibility; unused).
        split_enabled: legacy flag, ignored (per-fraction model has no split mechanism).
        augmecon_mode: switch standard eps-constraint to AUGMECON2 with slack.
        penalty_range, delta: AUGMECON2 hyperparameters.
    """
    model = pyo.ConcreteModel(name="Centrum_TriModal_PerFraction")

    # i -> [waters], [mobiles], [ucs] per fraction
    waters_for_hh  = {}
    mobiles_for_hh = {}
    ucs_for_hh_t   = {t: {} for t in WASTE_TYPES}
    for (i, w) in feas_w:
        waters_for_hh.setdefault(i, []).append(w)
    for (i, m) in feas_m:
        mobiles_for_hh.setdefault(i, []).append(m)
    for t in WASTE_TYPES:
        for (i, u) in feas_u_t[t]:
            ucs_for_hh_t[t].setdefault(i, []).append(u)

    # w -> [households], m -> [households], (t, u) -> [households]
    hh_for_water  = {}
    hh_for_mobile = {}
    hh_for_uc_t   = {t: {} for t in WASTE_TYPES}
    for (i, w) in feas_w:
        hh_for_water.setdefault(w, []).append(i)
    for (i, m) in feas_m:
        hh_for_mobile.setdefault(m, []).append(i)
    for t in WASTE_TYPES:
        for (i, u) in feas_u_t[t]:
            hh_for_uc_t[t].setdefault(u, []).append(i)

    # Flat UC list + reverse fraction lookup
    all_U = []
    uc_fraction = {}
    for t in WASTE_TYPES:
        for u in U_t[t]:
            all_U.append(u)
            uc_fraction[u] = t

    # --- SETS ---
    model.I = pyo.Set(initialize=I)
    model.W = pyo.Set(initialize=W)
    model.M = pyo.Set(initialize=M)
    model.U = pyo.Set(initialize=all_U)
    model.T = pyo.Set(initialize=WASTE_TYPES)

    # Per-fraction feasibility sets: (i, w, t) etc.
    feas_w_flat = [(i, w, t) for (i, w) in feas_w for t in WASTE_TYPES]
    feas_m_flat = [(i, m, t) for (i, m) in feas_m for t in WASTE_TYPES]
    feas_u_flat = [(i, t, u) for t in WASTE_TYPES for (i, u) in feas_u_t[t]]
    model.FeasWaterPairs  = pyo.Set(within=model.I * model.W * model.T, initialize=feas_w_flat)
    model.FeasMobilePairs = pyo.Set(within=model.I * model.M * model.T, initialize=feas_m_flat)
    model.FeasUgPairs     = pyo.Set(within=model.I * model.T * model.U, initialize=feas_u_flat)

    # --- DECISION VARIABLES ---
    # Facility activation (binary)
    model.y_w = pyo.Var(model.W, within=pyo.Binary, doc="1 if water location is open")
    model.y_m = pyo.Var(model.M, within=pyo.Binary, doc="1 if mobile location is open")
    model.y_u = pyo.Var(model.U, within=pyo.Binary, doc="1 if UC is activated")
    model.press = pyo.Var(model.U, within=pyo.Binary, doc="1 if UC is configured as press container")

    # Water emptying trips (integer)
    model.z_w = pyo.Var(model.W, within=pyo.NonNegativeIntegers, bounds=(0, 1000),
                        doc="Number of emptying trips for each water location")

    # Fleet (integer)
    model.U_shifts = pyo.Var(within=pyo.NonNegativeIntegers, doc="Number of daily fleet shifts")
    model.n_dumps  = pyo.Var(within=pyo.NonNegativeIntegers, doc="Number of daily replenishment trips")

    # Assignment fractions (continuous in [0, 1] — LP relaxation, see module docstring)
    model.x_w = pyo.Var(model.FeasWaterPairs,  within=pyo.NonNegativeReals, bounds=(0, 1),
                        doc="Fraction of hh i's fraction-t volume going to water w")
    model.x_m = pyo.Var(model.FeasMobilePairs, within=pyo.NonNegativeReals, bounds=(0, 1),
                        doc="Fraction of hh i's fraction-t volume going to mobile m")
    model.x_u = pyo.Var(model.FeasUgPairs,     within=pyo.NonNegativeReals, bounds=(0, 1),
                        doc="Fraction of hh i's fraction-t volume going to UC u")

    # --- AUGMECON SLACK ---
    if augmecon_mode and epsilon is not None:
        model.slack = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, epsilon),
                              doc="AUGMECON2 slack variable for the penalty constraint")
        if penalty_range is None or penalty_range <= 0:
            penalty_range = max(epsilon, 1.0)
        model.penalty_range = penalty_range
        model.delta = delta

    # --- OBJECTIVE EXPRESSIONS ---
    def calc_cost(model):
        water_fixed   = sum(params['f_W']       * model.y_w[w] for w in model.W)
        water_var     = sum(params['c_W_empty'] * model.z_w[w] for w in model.W)
        mobile_fleet  = model.U_shifts * params['D_week'] * params['C_shift']
        mobile_epsilon = sum(0.01 * model.y_m[m] for m in model.M)  # tiny regularizer
        uc_cost = sum(
            params['f_U'][uc_fraction[u]] * model.y_u[u]
            + (params['f_U_press'][uc_fraction[u]] - params['f_U'][uc_fraction[u]]) * model.press[u]
            for u in model.U
        )
        return water_fixed + water_var + mobile_fleet + mobile_epsilon + uc_cost
    model.TotalCostExpr = pyo.Expression(rule=calc_cost)

    # Per-fraction penalty weights (default = 1.0 each, i.e. thesis-style
    # symmetric penalty). Setting a weight to 0 turns off that fraction's
    # contribution to the walking-distance objective so only the hard R_max
    # constraint remains; intermediate values are useful for frequency-weighted
    # penalty variants.
    pw = params.get('penalty_weight', {t: 1.0 for t in WASTE_TYPES})

    # Walking penalty — summed across fractions; if a household walks to the same
    # facility for multiple fractions, that's counted once per fraction.
    def calc_penalty(model):
        pure_w = sum(pw.get(t, 1.0) * pen_w[i, w] * model.x_w[i, w, t]
                     for (i, w, t) in model.FeasWaterPairs)
        pure_m = sum(pw.get(t, 1.0) * pen_m[i, m] * model.x_m[i, m, t]
                     for (i, m, t) in model.FeasMobilePairs)
        pure_u = sum(pw.get(t, 1.0) * pen_u_t[t][i, u] * model.x_u[i, t, u]
                     for (i, t, u) in model.FeasUgPairs)
        return pure_w + pure_m + pure_u
    model.TotalPenaltyExpr = pyo.Expression(rule=calc_penalty)

    # Raw walking penalty (no mobile bias) — same shape, used for reporting
    def calc_walking(model):
        pure_w = sum(pw.get(t, 1.0) * pen_w[i, w] * model.x_w[i, w, t]
                     for (i, w, t) in model.FeasWaterPairs)
        pure_m = sum(pw.get(t, 1.0) * pen_m_raw[i, m] * model.x_m[i, m, t]
                     for (i, m, t) in model.FeasMobilePairs)
        pure_u = sum(pw.get(t, 1.0) * pen_u_t[t][i, u] * model.x_u[i, t, u]
                     for (i, t, u) in model.FeasUgPairs)
        return pure_w + pure_m + pure_u
    model.WalkingPenaltyExpr = pyo.Expression(rule=calc_walking)

    # --- OBJECTIVE FUNCTION ---
    if augmecon_mode and epsilon is not None:
        def obj_rule_augmecon(model):
            return model.TotalCostExpr - model.delta * (model.slack / model.penalty_range)
        model.Objective = pyo.Objective(rule=obj_rule_augmecon, sense=pyo.minimize)
    else:
        def obj_rule(model):
            return model.TotalCostExpr
        model.Objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- EPSILON CONSTRAINT ---
    # epsilon is exposed as a mutable Param so a Pareto sweep can build the model
    # once and update the bound between solves instead of rebuilding each time.
    if epsilon is not None:
        model.epsilon_param = pyo.Param(initialize=epsilon, mutable=True, within=pyo.NonNegativeReals)
        if augmecon_mode:
            def epsilon_constraint_augmecon(model):
                return model.TotalPenaltyExpr + model.slack == model.epsilon_param
            model.EpsilonConstraint = pyo.Constraint(rule=epsilon_constraint_augmecon)
        else:
            def epsilon_constraint_rule(model):
                return model.TotalPenaltyExpr <= model.epsilon_param
            model.EpsilonConstraint = pyo.Constraint(rule=epsilon_constraint_rule)

    # --- CONSTRAINTS ---

    # 1. Fleet Time Budget (unchanged)
    def fleet_size_rule(model):
        visits_needed = params['F_freq'] * sum(model.y_m[m] for m in model.M)
        stops_lost = model.n_dumps * params['stop_cost_dump']
        return model.U_shifts * params['S_max'] >= visits_needed + stops_lost
    model.FleetSize = pyo.Constraint(rule=fleet_size_rule)

    # 2. Global Fleet Capacity (Weekly) — mobile load only, no split term anymore
    def global_fleet_cap(model, t):
        total_vol_m = sum(demand[i, t] * model.x_m[i, m, t] for (i, m, _t) in model.FeasMobilePairs if _t == t)
        total_daily_trips = model.U_shifts + model.n_dumps
        return total_vol_m <= total_daily_trips * params['D_week'] * params['V_truck'][t]
    model.GlobalFleetCap = pyo.Constraint(model.T, rule=global_fleet_cap)

    # 3. Per-fraction strategy selection: each (i, t) is fully served (sum = 1)
    def strategy_per_fraction_rule(model, i, t):
        w_sum = sum(model.x_w[i, w, t] for w in waters_for_hh.get(i, []))
        m_sum = sum(model.x_m[i, m, t] for m in mobiles_for_hh.get(i, []))
        u_sum = sum(model.x_u[i, t, u] for u in ucs_for_hh_t[t].get(i, []))
        return w_sum + m_sum + u_sum == 1
    model.StrategyPerFraction = pyo.Constraint(model.I, model.T, rule=strategy_per_fraction_rule)

    # 4. Linking: assignment requires facility / UC to be open
    def link_water_rule(model, i, w, t):
        return model.x_w[i, w, t] <= model.y_w[w]
    model.LinkWater = pyo.Constraint(model.FeasWaterPairs, rule=link_water_rule)

    def link_mobile_rule(model, i, m, t):
        return model.x_m[i, m, t] <= model.y_m[m]
    model.LinkMobile = pyo.Constraint(model.FeasMobilePairs, rule=link_mobile_rule)

    def link_ug_rule(model, i, t, u):
        return model.x_u[i, t, u] <= model.y_u[u]
    model.LinkUg = pyo.Constraint(model.FeasUgPairs, rule=link_ug_rule)

    # 5. Press configuration: press[u] only if y_u[u] = 1
    def press_link_rule(model, u):
        return model.press[u] <= model.y_u[u]
    model.PressLink = pyo.Constraint(model.U, rule=press_link_rule)

    # 5b. Global press toggle: if disabled, force press[u] = 0
    if not params.get('press_enabled', True):
        def press_off_rule(model, u):
            return model.press[u] == 0
        model.PressOff = pyo.Constraint(model.U, rule=press_off_rule)

    # 5c. Glass UCs cannot be press containers (compaction not used for glass)
    glass_ucs = list(U_t.get('glass', []))
    if glass_ucs:
        model.GlassNoPress = pyo.Set(initialize=glass_ucs)
        def glass_no_press_rule(model, u):
            return model.press[u] == 0
        model.GlassNoPressC = pyo.Constraint(model.GlassNoPress, rule=glass_no_press_rule)

    # 6. Water Capacity (per location, per fraction) — no split term
    def cap_water(model, w, t):
        load = sum(demand[i, t] * model.x_w[i, w, t] for i in hh_for_water.get(w, []))
        return load <= params['C_W'][t] * model.z_w[w]
    model.CapacityWater = pyo.Constraint(model.W, model.T, rule=cap_water)

    # 7. Mobile Capacity (per location, per fraction) — no split term
    def cap_mobile(model, m, t):
        load = sum(demand[i, t] * model.x_m[i, m, t] for i in hh_for_mobile.get(m, []))
        return load <= params['C_M'][t] * model.y_m[m]
    model.CapacityMobile = pyo.Constraint(model.M, model.T, rule=cap_mobile)

    # 8. Underground Container Capacity (per UC, with press boost)
    # load(u) <= C_U[t(u)] * y_u[u] + (C_U_press[t(u)] - C_U[t(u)]) * press[u]
    def cap_ug_rule(model, u):
        t = uc_fraction[u]
        load = sum(demand[i, t] * model.x_u[i, t, u] for i in hh_for_uc_t[t].get(u, []))
        regular_cap = params['C_U'][t]
        press_extra = params['C_U_press'][t] - params['C_U'][t]
        return load <= regular_cap * model.y_u[u] + press_extra * model.press[u]
    model.CapacityUg = pyo.Constraint(model.U, rule=cap_ug_rule)

    return model
