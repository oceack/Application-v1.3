# backend/diagnostics.py
# This module is used to identify issues when infeasibility occurs
# It uses a relaxed model with slack variables to pinpoint:
# - Unreachable households
# - Capacity overflows
# - Fleet shortages
# Objective is changed to minimise the slack variables

"""
Diagnostic tool for identifying infeasibility causes.

Uses a relaxed model with slack variables to pinpoint:
- Unreachable households
- Capacity overflows
- Fleet shortages

Exports:
    run_diagnostics(...)
"""

import pyomo.environ as pyo

from .data_loader import load_data


def run_diagnostics(params, data_files, solver_name):
    """
    Runs a relaxed version of the model to identify why it is infeasible.
    
    Args:
        params: dict of model parameters
        data_files: dict with keys 'demand', 'water', 'mobile', 'coords'
        solver_name: 'gurobi', 'cplex', 'cbc', etc.

    Returns:
        (error_string, (report_dict, unreachable_list)) or (error_string, None)
    """
    # 1. Load Data
    data, error = load_data(
        data_files['demand'], data_files['water'], data_files['mobile'], 
        data_files.get('coords'), params
    )
    if error: 
        return f"Data Error: {error}", None
    
    (I, W, M, U, feas_w, feas_m, feas_u, demand, pen_w, pen_m, _, pen_u, _, _, _, _) = data
    
    # Pre-check reachability (including underground containers)
    covered = set(i for (i, w) in feas_w) | set(i for (i, m) in feas_m) | set(i for (i, u) in feas_u)
    totally_unreachable = [i for i in I if i not in covered]

    # Build Relaxed Model
    model = pyo.ConcreteModel(name="Diagnostic")
    model.I = pyo.Set(initialize=I)
    model.W = pyo.Set(initialize=W)
    model.M = pyo.Set(initialize=M)
    model.U = pyo.Set(initialize=U)
    model.T = pyo.Set(initialize=['rest', 'cardboard', 'glass'])
    model.FeasWaterPairs = pyo.Set(within=model.I * model.W, initialize=feas_w)
    model.FeasMobilePairs = pyo.Set(within=model.I * model.M, initialize=feas_m)
    model.FeasUndergroundPairs = pyo.Set(within=model.I * model.U, initialize=feas_u)

    # Slack Variables ("Cheats") for each I/W/M for each t
    model.slack_unserved = pyo.Var(model.I, within=pyo.NonNegativeReals)
    model.slack_cap_w = pyo.Var(model.W, model.T, within=pyo.NonNegativeReals)
    model.slack_cap_m = pyo.Var(model.M, model.T, within=pyo.NonNegativeReals)
    model.slack_fleet = pyo.Var(within=pyo.NonNegativeReals)
    model.slack_global_cap = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # Real Variables
    model.y_w = pyo.Var(model.W, within=pyo.Binary)
    model.y_m = pyo.Var(model.M, within=pyo.Binary)
    model.z_w = pyo.Var(model.W, within=pyo.NonNegativeIntegers, bounds=(0, 1000))
    model.U_shifts = pyo.Var(within=pyo.NonNegativeIntegers)
    model.n_dumps = pyo.Var(within=pyo.NonNegativeIntegers)

    model.x_w = pyo.Var(model.FeasWaterPairs, within=pyo.Binary)
    model.x_m = pyo.Var(model.FeasMobilePairs, within=pyo.Binary)
    model.x_u = pyo.Var(model.FeasUndergroundPairs, within=pyo.Binary)
    model.s_m = pyo.Var(model.FeasMobilePairs, within=pyo.Binary)
    model.r_w = pyo.Var(model.FeasWaterPairs, within=pyo.NonNegativeReals, bounds=(0, 1))

    # 1. Strategy Relaxed
    def strat_relax(m, i):
        return (
            sum(m.x_w[i, w] for w in m.W if (i, w) in m.FeasWaterPairs) +
            sum(m.x_m[i, mob] for mob in m.M if (i, mob) in m.FeasMobilePairs) +
            sum(m.x_u[i, u] for u in m.U if (i, u) in m.FeasUndergroundPairs) +
            sum(m.s_m[i, mob] for mob in m.M if (i, mob) in m.FeasMobilePairs) +
            m.slack_unserved[i] == 1
        )
    model.Strategy = pyo.Constraint(model.I, rule=strat_relax)

    # 2. Fleet Relaxed
    def fleet_relax(m):
        visits = params['F_freq'] * sum(m.y_m[mob] for mob in m.M)
        stops_lost = m.n_dumps * params['stop_cost_dump']
        return m.U_shifts * params['S_max'] >= visits + stops_lost - m.slack_fleet
    model.Fleet = pyo.Constraint(rule=fleet_relax)

    # 3. Global Capacity Relaxed
    def global_cap_relax(m, t):
        total_vol_m = sum(
            (demand[i, t] * m.x_m[i, mob]) + 
            (demand[i, t] * (1 - params['beta'][t]) * m.s_m[i, mob])
            for (i, mob) in m.FeasMobilePairs
        )
        total_daily_trips = m.U_shifts + m.n_dumps
        total_weekly_capacity = total_daily_trips * params['D_week'] * params['V_truck'][t]
        return total_vol_m <= total_weekly_capacity + m.slack_global_cap[t]
    model.GlobalFleetCap = pyo.Constraint(model.T, rule=global_cap_relax)

    # 4. Water Capacity Relaxed
    def cap_w_relax(m, w, t):
        load = sum(
            (demand[i, t] * m.x_w[i, w]) + (demand[i, t] * params['beta'][t] * m.r_w[i, w]) 
            for i in m.I if (i, w) in m.FeasWaterPairs
        )
        return load <= params['C_W'][t] * m.z_w[w] + m.slack_cap_w[w, t]
    model.CapW = pyo.Constraint(model.W, model.T, rule=cap_w_relax)

    # 5. Mobile Capacity Relaxed
    def cap_m_relax(m, mob, t):
        load = sum(
            (demand[i, t] * m.x_m[i, mob]) + (demand[i, t] * (1 - params['beta'][t]) * m.s_m[i, mob]) 
            for i in m.I if (i, mob) in m.FeasMobilePairs
        )
        return load <= params['C_M'][t] * m.y_m[mob] + m.slack_cap_m[mob, t]
    model.CapM = pyo.Constraint(model.M, model.T, rule=cap_m_relax)

    # Standard Links
    model.Residual = pyo.Constraint(
        model.I, 
        rule=lambda m, i: (
            sum(m.r_w[i, w] for w in m.W if (i, w) in m.FeasWaterPairs) == 
            sum(m.s_m[i, mob] for mob in m.M if (i, mob) in m.FeasMobilePairs)
        )
    )
    model.LinkW = pyo.Constraint(
        model.FeasWaterPairs, 
        rule=lambda m, i, w: m.x_w[i, w] + m.r_w[i, w] <= m.y_w[w]
    )
    model.LinkM = pyo.Constraint(
        model.FeasMobilePairs, 
        rule=lambda m, i, mob: m.x_m[i, mob] + m.s_m[i, mob] <= m.y_m[mob]
    )
    model.LinkTrips = pyo.Constraint(
        model.W, 
        rule=lambda m, w: m.z_w[w] <= 1000 * m.y_w[w]
    )

    # Objective: Minimize Violations
    model.Obj = pyo.Objective(expr=(
        sum(model.slack_unserved[i] for i in model.I) * 1000 + 
        sum(model.slack_cap_w[w, t] for w in model.W for t in model.T) + 
        sum(model.slack_cap_m[mob, t] for mob in model.M for t in model.T) + 
        sum(model.slack_global_cap[t] for t in model.T) * 500 +
        model.slack_fleet * 100
    ))

    solver = pyo.SolverFactory(solver_name)
    try:
        solver.solve(model)
        
        total_global_overflow = pyo.value(sum(model.slack_global_cap[t] for t in model.T))
        
        report = {
            "Totally Unreachable (Distance)": len(totally_unreachable),
            "Unserved Households (Optimization)": sum(
                1 for i in model.I if pyo.value(model.slack_unserved[i]) > 0.01
            ),
            "Mobile Global Cap Overflow (L)": total_global_overflow,
            "Mobile Capacity Overflow (L)": pyo.value(
                sum(model.slack_cap_m[mob, t] for mob in model.M for t in model.T)
            ),
            "Water Capacity Overflow (L)": pyo.value(
                sum(model.slack_cap_w[w, t] for w in model.W for t in model.T)
            ),
            "Missing Shifts (Fleet Constraint)": pyo.value(model.slack_fleet) / params['S_max']
        }
        return None, (report, totally_unreachable[:10])
    except Exception as e:
        return f"Diagnostic Failed: {e}", None
