# backend/solver.py
"""
Optimization loop and result extraction.

Exports:
    run_optimization(...)  - Standard epsilon-constraint method (backward compatible)
    run_augmecon(...)      - AUGMECON2 method with payoff table and acceleration
    compute_payoff_table(...) - Compute objective ranges for grid generation
"""

import pyomo.environ as pyo
import pandas as pd
import time

from .data_loader import load_data
from .model_builder import build_fleet_model
from .parameters import AUGMECON_DEFAULTS


def run_optimization(params, data_files, epsilon_values, solver_name, progress_bar, time_limit=None, mip_gap=None):
    """
    Main optimization entry point using epsilon-constraint method.

    Args:
        params: dict of model parameters
        data_files: dict with keys 'demand', 'water', 'mobile', 'coords'
        epsilon_values: list of epsilon floats to sweep (penalty constraint bounds)
        solver_name: 'gurobi', 'cplex', 'cbc', etc.
        progress_bar: Streamlit progress bar or None

    Returns:
        (figure, results_df, detailed_results, allocations_results, distance_distributions)
    """
    # 1. Load Data
    data, error = load_data(
        data_files['demand'], data_files['water'], data_files['mobile'], 
        data_files.get('coords'), params
    )
    if error: 
        raise Exception(error)
    
    # unpack data
    (I, W, M, U, feas_w, feas_m, feas_u, demand, pen_w, pen_m, pen_m_raw, pen_u,
     coord_map, dist_water, dist_mobile, dist_underground) = data
    
    results = []
    detailed_results = {} 
    allocations_results = {}
    distance_distributions = {}  # Store distance distribution per epsilon
    
    solver = pyo.SolverFactory(solver_name)

    # Apply optional limits per solver
    if time_limit is not None:
        if solver_name == 'gurobi':
            solver.options['TimeLimit'] = time_limit
        elif solver_name == 'cplex':
            solver.options['timelimit'] = time_limit
        elif solver_name == 'cbc':
            solver.options['seconds'] = time_limit
        elif solver_name == 'glpk':
            solver.options['tmlim'] = time_limit
        elif solver_name == 'scip':
            solver.options['limits/time'] = time_limit

    if mip_gap is not None:
        if solver_name == 'gurobi':
            solver.options['MIPGap'] = mip_gap
        elif solver_name == 'cplex':
            solver.options['mipgap'] = mip_gap
        elif solver_name == 'cbc':
            solver.options['ratioGap'] = mip_gap
        elif solver_name == 'glpk':
            solver.options['mipgap'] = mip_gap
        elif solver_name == 'scip':
            solver.options['limits/gap'] = mip_gap
    
    # 2. Iterate Epsilon values and check if split is enabled
    split_enabled = params.get('split_strategy_enabled', True)
    
    for idx, epsilon in enumerate(epsilon_values):
        model = build_fleet_model(
            I, W, M, U, feas_w, feas_m, feas_u, demand, params, pen_w, pen_m, pen_m_raw, pen_u, epsilon=epsilon,
            dist_water=dist_water, dist_mobile=dist_mobile, split_enabled=split_enabled
        )
        
        try:
            start_time = time.time()
            res = solver.solve(model, tee=False, symbolic_solver_labels=False)
            solve_time = time.time() - start_time
            
            # Check for optimal OR time limit with feasible solution
            term_cond = res.solver.termination_condition
            if term_cond == pyo.TerminationCondition.optimal or term_cond == pyo.TerminationCondition.maxTimeLimit:
                cost = pyo.value(model.TotalCostExpr)
                walking_penalty = pyo.value(model.WalkingPenaltyExpr)
                

                # Extract solver stats
                termination_status = "Optimal"
                try:
                    gap = res.solver.termination_condition
                    # Because of all the different solver, the gap is not always available
                    if hasattr(res, 'problem'):
                        if hasattr(res.problem, 'upper_bound') and hasattr(res.problem, 'lower_bound'):
                            lb = res.problem.lower_bound
                            ub = res.problem.upper_bound
                            if lb is not None and ub is not None and abs(lb) > 1e-6:
                                gap = abs((ub - lb) / lb)
                            else:
                                gap = 0.0
                        else:
                            gap = 0.0
                    else:
                        gap = 0.0
                except:
                    gap = 0.0
                
                # --- RESULTS PROCESSING ---
                load_w = {w: 0.0 for w in model.W}
                load_m = {m: 0.0 for m in model.M}
                load_u = {u: 0.0 for u in model.U}
                client_count_w = {w: 0 for w in model.W}
                client_count_m = {m: 0 for m in model.M}
                client_count_u = {u: 0 for u in model.U}
                
                total_dist_sum = 0  # Sum of distances (unweighted)
                total_count = 0      # Count of household assignments
                current_max_dist = 0
                
                alloc_list = []
                
                # Process Water Assignments
                for (i, w) in model.FeasWaterPairs:
                    dist_val = dist_water.get((i, w), 0)
                    hh_lat, hh_lon = coord_map.get(i, (None, None))
                    used = False
                    
                    if pyo.value(model.x_w[i, w]) > 0.5:
                        used = True
                        alloc_list.append({
                            'Household': i, 'Facility': w, 'Type': 'Water', 
                            'Strategy': 'Pure', 'Distance': dist_val,
                            'hh_lat': hh_lat, 'hh_lon': hh_lon
                        })
                    
                    # Check for auto-split load at this water (from mobile users nearby)
                    split_load_here = 0
                    if split_enabled and hasattr(model, 'split_at_water') and hasattr(model, 'SplitWaterPairs') and (i, w) in model.SplitWaterPairs:
                        if pyo.value(model.split_at_water[i, w]) > 0.5:
                            # This mobile user also uses this water
                            alloc_list.append({
                                'Household': i, 'Facility': w, 'Type': 'Water', 
                                'Strategy': 'AutoSplit', 'Distance': dist_val,
                                'hh_lat': hh_lat, 'hh_lon': hh_lon
                            })
                            used = True
                    
                    if used:
                        client_count_w[w] += 1
                        total_dist_sum += dist_val
                        total_count += 1
                        if dist_val > current_max_dist: 
                            current_max_dist = dist_val
                        for t in model.T:
                            split_val = 0
                            if split_enabled and hasattr(model, 'split_at_water') and hasattr(model, 'SplitWaterPairs') and (i, w) in model.SplitWaterPairs:
                                split_val = pyo.value(model.split_at_water[i, w])
                            val = (
                                (demand[i, t] * pyo.value(model.x_w[i, w])) +
                                (demand[i, t] * params['beta'][t] * split_val)
                            )
                            load_w[w] += val
                        
                # Process Mobile Assignments
                for (i, m) in model.FeasMobilePairs:
                    dist_val = dist_mobile.get((i, m), 0)
                    hh_lat, hh_lon = coord_map.get(i, (None, None))
                    used = False
                    
                    if pyo.value(model.x_m[i, m]) > 0.5:
                        used = True
                        # Check if this mobile user auto-splits with water
                        is_split = False
                        if split_enabled and hasattr(model, 'split_at_mobile') and hasattr(model, 'SplitMobilePairs') and (i, m) in model.SplitMobilePairs:
                            is_split = pyo.value(model.split_at_mobile[i, m]) > 0.5
                        
                        alloc_list.append({
                            'Household': i, 'Facility': m, 'Type': 'Mobile', 
                            'Strategy': 'AutoSplit' if is_split else 'Pure', 'Distance': dist_val,
                            'hh_lat': hh_lat, 'hh_lon': hh_lon
                        })
                        
                    if used:
                        client_count_m[m] += 1
                        total_dist_sum += dist_val
                        total_count += 1
                        if dist_val > current_max_dist: 
                            current_max_dist = dist_val
                        for t in model.T:
                            split_val = 0
                            if split_enabled and hasattr(model, 'split_at_mobile') and hasattr(model, 'SplitMobilePairs') and (i, m) in model.SplitMobilePairs:
                                split_val = pyo.value(model.split_at_mobile[i, m])
                            # If splitting, mobile only gets (1-beta) fraction of pure load
                            # pure load = demand * x_m, split reduces by beta * demand * split_at_mobile
                            val = demand[i, t] * pyo.value(model.x_m[i, m]) - demand[i, t] * params['beta'][t] * split_val
                            load_m[m] += val

                # Process Underground Assignments
                for (i, u) in model.FeasUndergroundPairs:
                    dist_val = dist_underground.get((i, u), 0)
                    hh_lat, hh_lon = coord_map.get(i, (None, None))
                    
                    if pyo.value(model.x_u[i, u]) > 0.5:
                        alloc_list.append({
                            'Household': i, 'Facility': u, 'Type': 'Underground', 
                            'Strategy': 'Pure', 'Distance': dist_val,
                            'hh_lat': hh_lat, 'hh_lon': hh_lon
                        })
                        client_count_u[u] += 1
                        total_dist_sum += dist_val
                        total_count += 1
                        if dist_val > current_max_dist: 
                            current_max_dist = dist_val
                        for t in model.T:
                            val = demand[i, t] * pyo.value(model.x_u[i, u])
                            load_u[u] += val

                avg_dist = total_dist_sum / total_count if total_count > 0 else 0
                allocations_results[epsilon] = pd.DataFrame(alloc_list)
                

                # --- NEW: Compute distance distribution for histogram ---
                # Get distances per household (take the distance from allocation)
                alloc_df_temp = pd.DataFrame(alloc_list)
                if not alloc_df_temp.empty:
                    # Group by household to get the effective distance per household
                    # A household might have multiple allocations (split strategy), use the max distance
                    household_distances = alloc_df_temp.groupby('Household').agg({
                        'Distance': 'max',
                        'Type': 'first'  # Primary assignment type
                    }).reset_index()
                    
                    # Separate water, mobile, and underground assignments for histogram
                    water_assignments = alloc_df_temp[alloc_df_temp['Type'] == 'Water'].copy()
                    mobile_assignments = alloc_df_temp[alloc_df_temp['Type'] == 'Mobile'].copy()
                    underground_assignments = alloc_df_temp[alloc_df_temp['Type'] == 'Underground'].copy()
                    
                    distance_distributions[epsilon] = {
                        'all_distances': household_distances['Distance'].tolist(),
                        'water_distances': water_assignments['Distance'].tolist() if not water_assignments.empty else [],
                        'mobile_distances': mobile_assignments['Distance'].tolist() if not mobile_assignments.empty else [],
                        'underground_distances': underground_assignments['Distance'].tolist() if not underground_assignments.empty else [],
                        'allocations': alloc_df_temp
                    }
                else:
                    distance_distributions[epsilon] = {
                        'all_distances': [],
                        'water_distances': [],
                        'mobile_distances': [],
                        'underground_distances': [],
                        'allocations': pd.DataFrame()
                    }

                # Generate Facility List
                epsilon_facilities = []
                for w in model.W:
                    if pyo.value(model.y_w[w]) > 0.5:
                        trips = pyo.value(model.z_w[w])
                        lat, lon = coord_map.get(w, (None, None))
                        epsilon_facilities.append({
                            'Epsilon': epsilon, 'Type': 'Water', 'ID': w, 
                            'Capacity_Units': trips, 'Allocated_Demand': load_w[w],
                            'Clients': client_count_w[w],
                            'lat': lat, 'lon': lon
                        })
                for m in model.M:
                    if pyo.value(model.y_m[m]) > 0.5:
                        lat, lon = coord_map.get(m, (None, None))
                        epsilon_facilities.append({
                            'Epsilon': epsilon, 'Type': 'Mobile', 'ID': m, 
                            'Capacity_Units': 1, 'Allocated_Demand': load_m[m],
                            'Clients': client_count_m[m],
                            'lat': lat, 'lon': lon
                        })
                
                # Add Underground containers (always "open" if they have clients)
                for u in model.U:
                    if client_count_u[u] > 0:
                        lat, lon = coord_map.get(u, (None, None))
                        epsilon_facilities.append({
                            'Epsilon': epsilon, 'Type': 'Underground', 'ID': u, 
                            'Capacity_Units': 1, 'Allocated_Demand': load_u[u],
                            'Clients': client_count_u[u],
                            'lat': lat, 'lon': lon
                        })

                detailed_results[epsilon] = pd.DataFrame(epsilon_facilities)
                
                results.append({
                    "Epsilon": epsilon, 
                    "Cost": cost, 
                    "Total_Penalty": walking_penalty, 
                    "Avg_Distance": avg_dist,
                    "Max_Distance": current_max_dist,
                    "Boats": sum(pyo.value(model.y_w[w]) for w in model.W),
                    "Mobile_Locs": sum(pyo.value(model.y_m[m]) for m in model.M),
                    "Underground_Used": sum(1 for u in model.U if client_count_u[u] > 0),
                    "Shifts": pyo.value(model.U_shifts),
                    "Replenish_Dumps": pyo.value(model.n_dumps),
                    "Solve_Time_s": solve_time,
                    "Status": termination_status,
                    "Gap": gap
                })
            else:
                # Non-optimal termination (infeasible, unbounded, etc.)
                solve_time = time.time() - start_time
                status_str = str(res.solver.termination_condition)
                results.append({
                    "Epsilon": epsilon,
                    "Cost": None,
                    "Total_Penalty": None,
                    "Avg_Distance": None,
                    "Max_Distance": None,
                    "Boats": None,
                    "Mobile_Locs": None,
                    "Underground_Used": None,
                    "Shifts": None,
                    "Replenish_Dumps": None,
                    "Solve_Time_s": solve_time,
                    "Status": status_str,
                    "Gap": None
                })
        except Exception as e:
            results.append({
                "Epsilon": epsilon,
                "Cost": None,
                "Total_Penalty": None,
                "Avg_Distance": None,
                "Max_Distance": None,
                "Boats": None,
                "Mobile_Locs": None,
                "Underground_Used": None,
                "Shifts": None,
                "Replenish_Dumps": None,
                "Solve_Time_s": 0,
                "Status": f"Error: {str(e)[:50]}",
                "Gap": None
            })
        
        if progress_bar: 
            progress_bar.progress((idx + 1) / len(epsilon_values))

    return None, pd.DataFrame(results), detailed_results, allocations_results, distance_distributions


# ==========================================
# AUGMECON2 IMPLEMENTATION
# ==========================================

def compute_payoff_table(data, params, solver, time_limit=None, mip_gap=None):
    """
    Compute the payoff table to find the range of each objective.
    
    This solves two optimization problems:
    1. Minimize cost (no constraint) -> get penalty at this point
    2. Minimize penalty (no constraint) -> get cost at this point
    
    Args:
        data: tuple from load_data containing (I, W, M, U, feas_w, feas_m, ...)
        params: model parameters
        solver: Pyomo SolverFactory instance
        time_limit: optional time limit for each solve
        mip_gap: optional MIP gap
        
    Returns:
        dict with keys: cost_min, cost_at_penalty_min, penalty_min, penalty_at_cost_min
    """
    (I, W, M, U, feas_w, feas_m, feas_u, demand, pen_w, pen_m, pen_m_raw, pen_u,
     coord_map, dist_water, dist_mobile, dist_underground) = data
    
    split_enabled = params.get('split_strategy_enabled', True)
    
    # --- Solve 1: Minimize cost (unconstrained) ---
    model_cost = build_fleet_model(
        I, W, M, U, feas_w, feas_m, feas_u, demand, params, 
        pen_w, pen_m, pen_m_raw, pen_u, 
        epsilon=None,  # No constraint
        dist_water=dist_water, dist_mobile=dist_mobile,
        split_enabled=split_enabled,
        augmecon_mode=False
    )
    
    try:
        res_cost = solver.solve(model_cost, tee=False)
        term_cond = res_cost.solver.termination_condition
        if term_cond == pyo.TerminationCondition.optimal or term_cond == pyo.TerminationCondition.maxTimeLimit:
            cost_min = pyo.value(model_cost.TotalCostExpr)
            penalty_at_cost_min = pyo.value(model_cost.TotalPenaltyExpr)
        else:
            raise Exception(f"Could not minimize cost: {term_cond}")
    except Exception as e:
        raise
    
    # --- Solve 2: Minimize penalty (unconstrained) ---
    model_penalty = build_fleet_model(
        I, W, M, U, feas_w, feas_m, feas_u, demand, params, 
        pen_w, pen_m, pen_m_raw, pen_u, 
        epsilon=None,
        dist_water=dist_water, dist_mobile=dist_mobile,
        split_enabled=split_enabled,
        augmecon_mode=False
    )
    
    model_penalty.del_component(model_penalty.Objective)
    model_penalty.PenaltyObjective = pyo.Objective(expr=model_penalty.TotalPenaltyExpr, sense=pyo.minimize)
    
    try:
        res_penalty = solver.solve(model_penalty, tee=False)
        term_cond = res_penalty.solver.termination_condition
        if term_cond == pyo.TerminationCondition.optimal or term_cond == pyo.TerminationCondition.maxTimeLimit:
            penalty_min = pyo.value(model_penalty.TotalPenaltyExpr)
            cost_at_penalty_min = pyo.value(model_penalty.TotalCostExpr)
        else:
            raise Exception(f"Could not minimize penalty: {term_cond}")
    except Exception as e:
        raise
    
    payoff = {
        'cost_min': cost_min,
        'cost_at_penalty_min': cost_at_penalty_min,
        'penalty_min': penalty_min,
        'penalty_at_cost_min': penalty_at_cost_min,
        'penalty_range': penalty_at_cost_min - penalty_min,
        'cost_range': cost_at_penalty_min - cost_min,
    }
    
    return payoff


def run_augmecon(params, data_files, n_grid_points, solver_name, progress_bar, 
                 time_limit=None, mip_gap=None, delta=None, early_exit=None):
    """
    Run AUGMECON2 algorithm to generate Pareto frontier.
    
    This method:
    1. Computes payoff table to find objective ranges
    2. Generates epsilon grid automatically
    3. Uses acceleration to skip infeasible regions
    4. Guarantees proper Pareto optimal points (not weakly Pareto)
    
    Args:
        params: dict of model parameters
        data_files: dict with keys 'demand', 'water', 'mobile', 'coords'
        n_grid_points: number of grid intervals (generates n+1 points)
        solver_name: 'gurobi', 'cplex', 'cbc', etc.
        progress_bar: Streamlit progress bar or None
        time_limit: optional time limit per solve
        mip_gap: optional MIP gap
        delta: augmentation coefficient (default from AUGMECON_DEFAULTS)
        early_exit: stop after consecutive infeasibles (default from AUGMECON_DEFAULTS)

    Returns:
        (figure, results_df, detailed_results, allocations_results, distance_distributions)
    """
    # Apply defaults
    if delta is None:
        delta = AUGMECON_DEFAULTS['delta']
    # early_exit makes sure that when an epsilon is infeasible, it stops trying the next epsilons
    if early_exit is None:
        early_exit = AUGMECON_DEFAULTS['early_exit']
    bypass_threshold = AUGMECON_DEFAULTS['bypass_threshold']
    
    # 1. Load Data
    data, error = load_data(
        data_files['demand'], data_files['water'], data_files['mobile'], 
        data_files.get('coords'), params
    )
    if error: 
        raise Exception(error)
    
    (I, W, M, U, feas_w, feas_m, feas_u, demand, pen_w, pen_m, pen_m_raw, pen_u,
     coord_map, dist_water, dist_mobile, dist_underground) = data
    
    # Create solver
    solver = pyo.SolverFactory(solver_name)
    
    # Apply optional limits per solver
    if time_limit is not None:
        if solver_name == 'gurobi':
            solver.options['TimeLimit'] = time_limit
        elif solver_name == 'cplex':
            solver.options['timelimit'] = time_limit
        elif solver_name == 'cbc':
            solver.options['seconds'] = time_limit
        elif solver_name == 'glpk':
            solver.options['tmlim'] = time_limit
        elif solver_name == 'scip':
            solver.options['limits/time'] = time_limit

    if mip_gap is not None:
        if solver_name == 'gurobi':
            solver.options['MIPGap'] = mip_gap
        elif solver_name == 'cplex':
            solver.options['mipgap'] = mip_gap
        elif solver_name == 'cbc':
            solver.options['ratioGap'] = mip_gap
        elif solver_name == 'glpk':
            solver.options['mipgap'] = mip_gap
        elif solver_name == 'scip':
            solver.options['limits/gap'] = mip_gap
    
    # 2. Compute Payoff Table
    payoff = compute_payoff_table(data, params, solver, time_limit, mip_gap)
    
    penalty_range = payoff['penalty_range']
    penalty_min = payoff['penalty_min']
    penalty_max = payoff['penalty_at_cost_min']
    
    if penalty_range <= 0:
        penalty_range = 1.0  # Prevent division by zero
    
    # 3. Generate Epsilon Grid
    grid_step = penalty_range / n_grid_points
    epsilons = [penalty_min + k * grid_step for k in range(n_grid_points + 1)]
    # Sort descending > more efficient
    epsilons = sorted(epsilons, reverse=True)
    
    # 4. AUGMECON2 Loop with Acceleration
    results = []
    detailed_results = {} 
    allocations_results = {}
    distance_distributions = {}
    
    split_enabled = params.get('split_strategy_enabled', True)
    
    bypass = 0  # Acceleration: skip counter
    consecutive_infeasible = 0  # For early exit
    
    for idx, epsilon in enumerate(epsilons):
        # uses continue to skip the rest of the loop until bypass is 0
        if bypass > 0:
            bypass -= 1
            if progress_bar:
                progress_bar.progress((idx + 1) / len(epsilons))
            continue
        
        model = build_fleet_model(
            I, W, M, U, feas_w, feas_m, feas_u, demand, params, 
            pen_w, pen_m, pen_m_raw, pen_u, 
            epsilon=epsilon,
            dist_water=dist_water, dist_mobile=dist_mobile,
            split_enabled=split_enabled,
            augmecon_mode=True,
            penalty_range=penalty_range,
            delta=delta
        )
        
        try:
            start_time = time.time()
            res = solver.solve(model, tee=False, symbolic_solver_labels=False)
            solve_time = time.time() - start_time
            
            term_cond = res.solver.termination_condition
            if term_cond == pyo.TerminationCondition.optimal or term_cond == pyo.TerminationCondition.maxTimeLimit:
                consecutive_infeasible = 0  # Reset early exit counter
                
                cost = pyo.value(model.TotalCostExpr)
                walking_penalty = pyo.value(model.WalkingPenaltyExpr)
                slack_val = pyo.value(model.slack)
                actual_penalty = pyo.value(model.TotalPenaltyExpr)
                

                
                # AUGMECON2 Acceleration: skip iterations if slack is large
                if slack_val > grid_step:
                    skip_count = int(slack_val / grid_step)
                    bypass = skip_count
                
                # --- Extract solution (reuse logic from run_optimization) ---
                result_row, facilities, alloc_list, dist_data = _extract_solution(
                    model, epsilon, cost, walking_penalty, solve_time,
                    demand, params, coord_map, dist_water, dist_mobile, dist_underground,
                    split_enabled
                )
                result_row['Slack'] = slack_val  # Add slack to results
                
                # Extract actual MIP gap from solver
                try:
                    if hasattr(res, 'problem') and hasattr(res.problem, 'lower_bound') and hasattr(res.problem, 'upper_bound'):
                        lb = res.problem.lower_bound
                        ub = res.problem.upper_bound
                        if lb is not None and ub is not None and abs(ub) > 1e-6:
                            result_row['Gap'] = abs((ub - lb) / ub)
                        else:
                            result_row['Gap'] = 0.0
                    else:
                        result_row['Gap'] = 0.0 if term_cond == pyo.TerminationCondition.optimal else None
                except:
                    result_row['Gap'] = 0.0 if term_cond == pyo.TerminationCondition.optimal else None
                
                # Update status based on termination
                result_row['Status'] = "Optimal" if term_cond == pyo.TerminationCondition.optimal else "TimeLimit"
                
                results.append(result_row)
                detailed_results[epsilon] = pd.DataFrame(facilities)
                allocations_results[epsilon] = pd.DataFrame(alloc_list)
                distance_distributions[epsilon] = dist_data
                
            else:
                # Infeasible
                consecutive_infeasible += 1
                
                results.append({
                    "Epsilon": epsilon,
                    "Cost": None,
                    "Total_Penalty": None,
                    "Slack": None,
                    "Avg_Distance": None,
                    "Max_Distance": None,
                    "Boats": None,
                    "Mobile_Locs": None,
                    "Underground_Used": None,
                    "Shifts": None,
                    "Replenish_Dumps": None,
                    "Solve_Time_s": time.time() - start_time,
                    "Status": str(term_cond),
                    "Gap": None
                })
                
                # Early exit check
                if early_exit and consecutive_infeasible >= bypass_threshold:
                    break
                    
        except Exception as e:
            results.append({
                "Epsilon": epsilon,
                "Cost": None,
                "Total_Penalty": None,
                "Slack": None,
                "Avg_Distance": None,
                "Max_Distance": None,
                "Boats": None,
                "Mobile_Locs": None,
                "Underground_Used": None,
                "Shifts": None,
                "Replenish_Dumps": None,
                "Solve_Time_s": 0,
                "Status": f"Error: {str(e)[:50]}",
                "Gap": None
            })
        
        if progress_bar:
            progress_bar.progress((idx + 1) / len(epsilons))
    
    
    return None, pd.DataFrame(results), detailed_results, allocations_results, distance_distributions


def _extract_solution(model, epsilon, cost, walking_penalty, solve_time,
                      demand, params, coord_map, dist_water, dist_mobile, dist_underground,
                      split_enabled):
    """
    Extract solution details from a solved model.
    Helper function shared by run_optimization and run_augmecon.
    """
    # Initialize accumulators
    load_w = {w: 0.0 for w in model.W}
    load_m = {m: 0.0 for m in model.M}
    load_u = {u: 0.0 for u in model.U}
    client_count_w = {w: 0 for w in model.W}
    client_count_m = {m: 0 for m in model.M}
    client_count_u = {u: 0 for u in model.U}
    
    total_dist_sum = 0
    total_count = 0
    current_max_dist = 0
    
    alloc_list = []
    
    # Process Water Assignments
    for (i, w) in model.FeasWaterPairs:
        dist_val = dist_water.get((i, w), 0)
        hh_lat, hh_lon = coord_map.get(i, (None, None))
        used = False
        
        if pyo.value(model.x_w[i, w]) > 0.5:
            used = True
            alloc_list.append({
                'Household': i, 'Facility': w, 'Type': 'Water', 
                'Strategy': 'Pure', 'Distance': dist_val,
                'hh_lat': hh_lat, 'hh_lon': hh_lon
            })
        
        # Check for auto-split load
        if split_enabled and hasattr(model, 'split_at_water') and hasattr(model, 'SplitWaterPairs') and (i, w) in model.SplitWaterPairs:
            if pyo.value(model.split_at_water[i, w]) > 0.5:
                alloc_list.append({
                    'Household': i, 'Facility': w, 'Type': 'Water', 
                    'Strategy': 'AutoSplit', 'Distance': dist_val,
                    'hh_lat': hh_lat, 'hh_lon': hh_lon
                })
                used = True
        
        if used:
            client_count_w[w] += 1
            total_dist_sum += dist_val
            total_count += 1
            if dist_val > current_max_dist: 
                current_max_dist = dist_val
            for t in model.T:
                split_val = 0
                if split_enabled and hasattr(model, 'split_at_water') and hasattr(model, 'SplitWaterPairs') and (i, w) in model.SplitWaterPairs:
                    split_val = pyo.value(model.split_at_water[i, w])
                val = (
                    (demand[i, t] * pyo.value(model.x_w[i, w])) +
                    (demand[i, t] * params['beta'][t] * split_val)
                )
                load_w[w] += val
                
    # Process Mobile Assignments
    for (i, m) in model.FeasMobilePairs:
        dist_val = dist_mobile.get((i, m), 0)
        hh_lat, hh_lon = coord_map.get(i, (None, None))
        used = False
        
        if pyo.value(model.x_m[i, m]) > 0.5:
            used = True
            is_split = False
            if split_enabled and hasattr(model, 'split_at_mobile') and hasattr(model, 'SplitMobilePairs') and (i, m) in model.SplitMobilePairs:
                is_split = pyo.value(model.split_at_mobile[i, m]) > 0.5
            
            alloc_list.append({
                'Household': i, 'Facility': m, 'Type': 'Mobile', 
                'Strategy': 'AutoSplit' if is_split else 'Pure', 'Distance': dist_val,
                'hh_lat': hh_lat, 'hh_lon': hh_lon
            })
            
        if used:
            client_count_m[m] += 1
            total_dist_sum += dist_val
            total_count += 1
            if dist_val > current_max_dist: 
                current_max_dist = dist_val
            for t in model.T:
                split_val = 0
                if split_enabled and hasattr(model, 'split_at_mobile') and hasattr(model, 'SplitMobilePairs') and (i, m) in model.SplitMobilePairs:
                    split_val = pyo.value(model.split_at_mobile[i, m])
                val = demand[i, t] * pyo.value(model.x_m[i, m]) - demand[i, t] * params['beta'][t] * split_val
                load_m[m] += val

    # Process Underground Assignments
    for (i, u) in model.FeasUndergroundPairs:
        dist_val = dist_underground.get((i, u), 0)
        hh_lat, hh_lon = coord_map.get(i, (None, None))
        
        if pyo.value(model.x_u[i, u]) > 0.5:
            alloc_list.append({
                'Household': i, 'Facility': u, 'Type': 'Underground', 
                'Strategy': 'Pure', 'Distance': dist_val,
                'hh_lat': hh_lat, 'hh_lon': hh_lon
            })
            client_count_u[u] += 1
            total_dist_sum += dist_val
            total_count += 1
            if dist_val > current_max_dist: 
                current_max_dist = dist_val
            for t in model.T:
                val = demand[i, t] * pyo.value(model.x_u[i, u])
                load_u[u] += val

    avg_dist = total_dist_sum / total_count if total_count > 0 else 0
    
    # Generate Facility List
    facilities = []
    for w in model.W:
        if pyo.value(model.y_w[w]) > 0.5:
            trips = pyo.value(model.z_w[w])
            lat, lon = coord_map.get(w, (None, None))
            facilities.append({
                'Epsilon': epsilon, 'Type': 'Water', 'ID': w, 
                'Capacity_Units': trips, 'Allocated_Demand': load_w[w],
                'Clients': client_count_w[w],
                'lat': lat, 'lon': lon
            })
    for m in model.M:
        if pyo.value(model.y_m[m]) > 0.5:
            lat, lon = coord_map.get(m, (None, None))
            facilities.append({
                'Epsilon': epsilon, 'Type': 'Mobile', 'ID': m, 
                'Capacity_Units': 1, 'Allocated_Demand': load_m[m],
                'Clients': client_count_m[m],
                'lat': lat, 'lon': lon
            })
    for u in model.U:
        if client_count_u[u] > 0:
            lat, lon = coord_map.get(u, (None, None))
            facilities.append({
                'Epsilon': epsilon, 'Type': 'Underground', 'ID': u, 
                'Capacity_Units': 1, 'Allocated_Demand': load_u[u],
                'Clients': client_count_u[u],
                'lat': lat, 'lon': lon
            })

    # Distance distribution data
    alloc_df_temp = pd.DataFrame(alloc_list)
    if not alloc_df_temp.empty:
        water_assignments = alloc_df_temp[alloc_df_temp['Type'] == 'Water'].copy()
        mobile_assignments = alloc_df_temp[alloc_df_temp['Type'] == 'Mobile'].copy()
        underground_assignments = alloc_df_temp[alloc_df_temp['Type'] == 'Underground'].copy()
        
        dist_data = {
            'all_distances': alloc_df_temp['Distance'].tolist(),
            'water_distances': water_assignments['Distance'].tolist() if not water_assignments.empty else [],
            'mobile_distances': mobile_assignments['Distance'].tolist() if not mobile_assignments.empty else [],
            'underground_distances': underground_assignments['Distance'].tolist() if not underground_assignments.empty else [],
            'allocations': alloc_df_temp
        }
    else:
        dist_data = {
            'all_distances': [], 'water_distances': [], 'mobile_distances': [],
            'underground_distances': [], 'allocations': pd.DataFrame()
        }
    
    result_row = {
        "Epsilon": epsilon, 
        "Cost": cost, 
        "Total_Penalty": walking_penalty, 
        "Avg_Distance": avg_dist,
        "Max_Distance": current_max_dist,
        "Boats": sum(pyo.value(model.y_w[w]) for w in model.W),
        "Mobile_Locs": sum(pyo.value(model.y_m[m]) for m in model.M),
        "Underground_Used": sum(1 for u in model.U if client_count_u[u] > 0),
        "Shifts": pyo.value(model.U_shifts),
        "Replenish_Dumps": pyo.value(model.n_dumps),
        "Solve_Time_s": solve_time,
        "Status": "Optimal",
        "Gap": 0.0 # gets overwritten in the loop later on
    }
    
    return result_row, facilities, alloc_list, dist_data
