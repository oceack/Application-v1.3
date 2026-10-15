import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# Try to import adjustText for non-overlapping labels
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False

# --- IMPORT FOLIUM FOR INTERACTIVE MAPS ---
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# --- IMPORT BACKEND ---
try:
    from backend import run_optimization, run_augmecon, run_diagnostics, DEFAULT_PARAMS, AUGMECON_DEFAULTS
except ImportError as e:
    st.error(f"CRITICAL ERROR: Could not import 'backend' package.\nDetails: {e}")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Amsterdam Waste Optimizer v1.2", layout="wide")
st.title("Waste Logistics: Multi-Objective Optimization (v1.2 + AUGMECON2)")

# --- SESSION STATE SETUP ---
if 'status' not in st.session_state: st.session_state['status'] = "input"
if 'opt_results' not in st.session_state: st.session_state['opt_results'] = None
if 'opt_details' not in st.session_state: st.session_state['opt_details'] = None 
if 'opt_allocations' not in st.session_state: st.session_state['opt_allocations'] = None
if 'opt_fig' not in st.session_state: st.session_state['opt_fig'] = None
if 'opt_distance_dist' not in st.session_state: st.session_state['opt_distance_dist'] = None
if 'diag_ran' not in st.session_state: st.session_state['diag_ran'] = False
if 'diag_results' not in st.session_state: st.session_state['diag_results'] = None
if 'error_msg' not in st.session_state: st.session_state['error_msg'] = ""
if 'selected_facility' not in st.session_state: st.session_state['selected_facility'] = None
if 'underground_containers' not in st.session_state: st.session_state['underground_containers'] = []

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Logo_Amsterdam_Gemeente.svg/1200px-Logo_Amsterdam_Gemeente.svg.png", width=100)
    
    st.header("1. Upload Data")
    data_files = {
        "demand": st.file_uploader("Demand Data (CSV)", type="csv"),
        "water": st.file_uploader("Water OD Matrix (CSV)", type="csv"),
        "mobile": st.file_uploader("Mobile OD Matrix (CSV)", type="csv"),
    }
    
    st.markdown("**(Optional) Coordinates for Map**")
    st.info("Upload files with `fid`, `X`, `Y` columns to see map results.")
    coord_files = {
        "demand": st.file_uploader("Households (Coords)", type="csv", key="c_dem"),
        "water": st.file_uploader("Water (Coords)", type="csv", key="c_wat"),
        "mobile": st.file_uploader("Mobile (Coords)", type="csv", key="c_mob")
    }
    data_files['coords'] = coord_files
    
    # --- UNDERGROUND CONTAINERS SECTION ---
    st.markdown("---")
    st.markdown("### 🏗️ Underground Containers")
    st.info("Existing underground containers (free to use, 250m range)")
    
    # Option to upload or click on map
    ug_input_method = st.radio(
        "Add containers by:",
        ["📍 Click on Map", "📁 Upload CSV"],
        horizontal=True,
        key="ug_input_method"
    )
    
    if ug_input_method == "📁 Upload CSV":
        underground_file = st.file_uploader(
            "Upload Underground Locations (CSV)", 
            type="csv", 
            key="underground_upload",
            help="CSV with columns: id, X, Y (or lat, lon)"
        )
        
        if underground_file is not None:
            try:
                ug_df = pd.read_csv(underground_file)
                if len(ug_df.columns) <= 1:
                    underground_file.seek(0)
                    ug_df = pd.read_csv(underground_file, sep=';')
                
                ug_df.columns = [c.lower() for c in ug_df.columns]
                
                # Find coordinate columns
                x_col = next((c for c in ug_df.columns if c in ['x', 'lon', 'longitude']), None)
                y_col = next((c for c in ug_df.columns if c in ['y', 'lat', 'latitude']), None)
                id_col = next((c for c in ug_df.columns if 'id' in c), None)
                
                if x_col and y_col:
                    containers = []
                    for idx, row in ug_df.iterrows():
                        container_id = f"UG-{row[id_col]}" if id_col else f"UG-{idx+1}"
                        containers.append({
                            'id': container_id,
                            'lat': row[y_col],
                            'lon': row[x_col]
                        })
                    st.session_state['underground_containers'] = containers
                    st.success(f"✅ Loaded {len(containers)} underground containers")
                else:
                    st.error("Could not find X/Y or lat/lon columns in the file")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    else:  # Click on Map
        st.caption("👆 Click on the map to add underground container locations")
        
        # Default center (Amsterdam)
        default_lat = 52.3676
        default_lon = 4.9041
        
        if FOLIUM_AVAILABLE:
            # Create a map for clicking
            ug_map = folium.Map(location=[default_lat, default_lon], zoom_start=14)
            
            # Add existing containers to map
            for container in st.session_state['underground_containers']:
                folium.Marker(
                    location=[container['lat'], container['lon']],
                    popup=container['id'],
                    icon=folium.Icon(color='green', icon='arrow-down', prefix='fa'),
                    tooltip=f"Click to remove: {container['id']}"
                ).add_to(ug_map)
            
            # Add instructions
            folium.LatLngPopup().add_to(ug_map)
            
            # Display map and capture clicks
            map_click_data = st_folium(
                ug_map, 
                width=280, 
                height=300, 
                key="underground_click_map"
            )
            
            # Handle click - add new container
            if map_click_data and map_click_data.get('last_clicked'):
                clicked_lat = map_click_data['last_clicked']['lat']
                clicked_lon = map_click_data['last_clicked']['lng']
                
                # Check if clicking near existing container (to remove)
                remove_idx = None
                for idx, container in enumerate(st.session_state['underground_containers']):
                    dist = ((container['lat'] - clicked_lat)**2 + (container['lon'] - clicked_lon)**2)**0.5
                    if dist < 0.0005:  # Very close click = remove
                        remove_idx = idx
                        break
                
                if remove_idx is not None:
                    removed = st.session_state['underground_containers'].pop(remove_idx)
                    st.toast(f"Removed {removed['id']}")
                    st.rerun()
                else:
                    # Add new container
                    new_id = f"UG-{len(st.session_state['underground_containers']) + 1}"
                    st.session_state['underground_containers'].append({
                        'id': new_id,
                        'lat': clicked_lat,
                        'lon': clicked_lon
                    })
                    st.toast(f"Added {new_id} at ({clicked_lat:.4f}, {clicked_lon:.4f})")
                    st.rerun()
            
            st.caption("🟢 = Existing container. Click empty area to add, click marker to remove.")
        else:
            st.warning("Install `folium` and `streamlit-folium` for map clicking feature")
            # Manual entry fallback
            col_lat, col_lon = st.columns(2)
            manual_lat = col_lat.number_input("Latitude", value=default_lat, format="%.6f", key="manual_lat")
            manual_lon = col_lon.number_input("Longitude", value=default_lon, format="%.6f", key="manual_lon")
            if st.button("➕ Add Container"):
                new_id = f"UG-{len(st.session_state['underground_containers']) + 1}"
                st.session_state['underground_containers'].append({
                    'id': new_id,
                    'lat': manual_lat,
                    'lon': manual_lon
                })
                st.rerun()
    
    # Show current underground containers
    if st.session_state['underground_containers']:
        with st.expander(f"📋 {len(st.session_state['underground_containers'])} Underground Containers"):
            ug_display = pd.DataFrame(st.session_state['underground_containers'])
            st.dataframe(ug_display, width='stretch', hide_index=True)
            
            if st.button("🗑️ Clear All Underground Containers"):
                st.session_state['underground_containers'] = []
                st.rerun()

    st.header("2. Solver Settings")
    solver_name = st.selectbox("Solver", ["gurobi", "cplex", "cbc", "scip", "glpk"])
    
    # --- AUGMECON2 Toggle ---
    st.markdown("### 🎯 Optimization Method")
    use_augmecon = st.checkbox(
        "Use AUGMECON2 (recommended)", 
        value=True,
        help="AUGMECON2 automatically finds the Pareto frontier with guaranteed proper Pareto points and acceleration."
    )
    
    if use_augmecon:
        st.success("✅ **AUGMECON2 Mode**: Automatic epsilon grid generation with acceleration")
        n_grid_points = st.slider(
            "Number of Pareto points", 
            min_value=3, max_value=50, value=AUGMECON_DEFAULTS['n_grid_points'],
            help="More points = finer Pareto frontier but longer solve time. Start with 10-15."
        )
        epsilons = []  # Not used in AUGMECON mode
        
        with st.expander("Advanced AUGMECON Settings"):
            augmecon_delta = st.number_input(
                "Delta (augmentation coefficient)", 
                value=AUGMECON_DEFAULTS['delta'], 
                min_value=1e-6, max_value=1e-1, 
                format="%.6f",
                help="Small value (10⁻³ to 10⁻⁶) added to ensure proper Pareto points."
            )
            augmecon_early_exit = st.checkbox(
                "Early exit on consecutive infeasibles", 
                value=AUGMECON_DEFAULTS['early_exit'],
                help="Stop early if multiple consecutive epsilon values are infeasible."
            )
    else:
        st.info("📋 **Manual Mode**: Specify epsilon values manually")
        st.markdown("**ε-Constraint Method**: Minimize cost subject to walking penalty ≤ ε")
        st.warning("💡 Start with LARGE ε values to find the feasible range. Lower ε = stricter constraint = may become infeasible.")
        epsilon_input = st.text_input(
            "Epsilon Values (ε)", 
            "10000000, 5000000, 2000000, 1000000, 500000, 200000, 100000, 50000",
            help="Comma-separated penalty constraint bounds. Start large, then reduce to explore trade-offs."
        )
        try: 
            epsilons = [float(x.strip()) for x in epsilon_input.split(',')]
            epsilons = sorted(epsilons, reverse=True)
        except: 
            epsilons = []
        n_grid_points = 0  # Not used in manual mode
        augmecon_delta = AUGMECON_DEFAULTS['delta']
        augmecon_early_exit = AUGMECON_DEFAULTS['early_exit']

    solver_time_limit = st.number_input(
        "Time limit (seconds)", value=3600, min_value=0,
        help="Maximum solve time per epsilon. Use longer times for more reliable results."
    )
    solver_mip_gap = st.number_input(
        "MIP gap (relative)", value=0.01, min_value=0.0, max_value=1.0, step=0.0001, format="%.4f",
        help="Relative optimality gap. 0.001 = 0.1%. Lower = more accurate but slower."
    )

    st.header("3. Parameters")
    
    if 'p' not in st.session_state:
        defaults = deepcopy(DEFAULT_PARAMS)
        st.session_state['p'] = {
            'R_W': defaults['R_W'],
            'R_W_ideal': defaults['R_W_ideal'],
            'R_M': defaults['R_M'],
            'R_M_ideal': defaults['R_M_ideal'],
            'f_W': defaults['f_W'],
            'c_W_empty': defaults['c_W_empty'],
            'C_shift': defaults['C_shift'],
            'S_max': defaults['S_max'],
            'F_freq': defaults['F_freq'],
            'D_week': defaults['D_week'],
            'beta_rest': defaults['beta']['rest'],
            'cw_rest': defaults['C_W']['rest'],
            'cm_rest': defaults['C_M']['rest'],
            'beta_card': defaults['beta']['cardboard'],
            'cw_card': defaults['C_W']['cardboard'],
            'cm_card': defaults['C_M']['cardboard'],
            'beta_glass': defaults['beta']['glass'],
            'cw_glass': defaults['C_W']['glass'],
            'cm_glass': defaults['C_M']['glass'],
            'stop_cost_dump': defaults['stop_cost_dump'],
            'v_truck_rest': defaults['V_truck']['rest'],
            'v_truck_card': defaults['V_truck']['cardboard'],
            'v_truck_glass': defaults['V_truck']['glass'],
            'p_bias': defaults['p_bias']
        }
    p = st.session_state['p']

    # --- Parameter Expanders ---
    
    with st.expander("A. Distances & Costs"):
        c1, c2 = st.columns(2)
        p['R_W'] = c1.number_input("Max Radius Water", value=p['R_W'])
        p['R_W_ideal'] = c2.number_input("Ideal Radius Water", value=p['R_W_ideal'])
        c1, c2 = st.columns(2)
        p['R_M'] = c1.number_input("Max Radius Mobile", value=p['R_M'])
        p['R_M_ideal'] = c2.number_input("Ideal Radius Mobile", value=p['R_M_ideal'])
        p['f_W'] = st.number_input("Fixed Cost Water", value=p['f_W'])
        p['c_W_empty'] = st.number_input("Var Cost Boat Trip", value=p['c_W_empty'])
        
        st.markdown("---")
        p['p_bias'] = st.number_input("Mobile Inconvenience Bias", value=p['p_bias'], step=500.0, 
                                      help="Penalty added to Mobile points. Higher = Stronger preference for Water.")
        
        map_bubble_size = st.slider("Map Bubble Size", min_value=5, max_value=100, value=15, 
                                    help="Adjust dot size on the result map.")

    with st.expander("B. Fleet Logistics"):
        p['C_shift'] = st.number_input("Cost/Shift", value=p['C_shift'])
        c1, c2 = st.columns(2)
        p['S_max'] = c1.number_input("Stops/Shift", value=p['S_max'])
        p['F_freq'] = c2.number_input("Visits/Day", value=p['F_freq'])
        p['D_week'] = st.number_input("Days/Week", value=p['D_week'])
        p['stop_cost_dump'] = st.number_input("Stops lost per Dump", value=p['stop_cost_dump'])

    with st.expander("C. Compartment Details"):
        # REST
        st.markdown("### Residual Waste")
        c1, c2, c3 = st.columns(3)
        p['beta_rest'] = c1.number_input("Beta (Rest)", value=p['beta_rest'], step=0.01)
        p['cw_rest'] = c2.number_input("Cap Water (Rest)", value=p['cw_rest'])
        p['cm_rest'] = c3.number_input("Cap Mob (Rest)", value=p['cm_rest'])
        p['v_truck_rest'] = st.number_input("Truck Vol (Rest)", value=p['v_truck_rest'])
        
        st.markdown("---")
        # CARDBOARD
        st.markdown("### Cardboard")
        c1, c2, c3 = st.columns(3)
        p['beta_card'] = c1.number_input("Beta (Card)", value=p['beta_card'], step=0.01)
        p['cw_card'] = c2.number_input("Cap Water (Card)", value=p['cw_card'])
        p['cm_card'] = c3.number_input("Cap Mob (Card)", value=p['cm_card'])
        p['v_truck_card'] = st.number_input("Truck Vol (Card)", value=p['v_truck_card'])

        st.markdown("---")
        # GLASS
        st.markdown("### Glass")
        c1, c2, c3 = st.columns(3)
        p['beta_glass'] = c1.number_input("Beta (Glass)", value=p['beta_glass'], step=0.01)
        p['cw_glass'] = c2.number_input("Cap Water (Glass)", value=p['cw_glass'])
        p['cm_glass'] = c3.number_input("Cap Mob (Glass)", value=p['cm_glass'])
        p['v_truck_glass'] = st.number_input("Truck Vol (Glass)", value=p['v_truck_glass'])

    # --- MOBILE RESTRICTION OPTION (Outside expander for visibility) ---
    st.markdown("---")
    st.markdown("### 🚛 Mobile Point Restriction")
    mobile_only_unreachable = st.checkbox(
        "Mobile only for unreachable households",
        value=False,
        key="mobile_only_unreachable_checkbox",
        help="When enabled, mobile points will ONLY be available for households that have NO waterpoint within the max water radius (R_W). Households covered by a waterpoint cannot use mobile points."
    )
    if mobile_only_unreachable:
        st.info("✅ Mobile restriction is ACTIVE: Only households without water coverage can use mobile points.")

    st.markdown("### 🔀 Automatic Split Strategy")
    split_strategy_enabled = st.checkbox(
        "Enable automatic split",
        value=False,
        key="split_strategy_checkbox",
        help="When enabled, households assigned to mobile will AUTOMATICALLY split with the nearest open water point (β goes to water, 1-β stays at mobile). Disable for faster solving."
    )
    if split_strategy_enabled:
        st.info("✅ Automatic split ENABLED: Mobile users will share load with nearby open water points.")
        force_split_enabled = False  # No longer used in automatic split model
    else:
        st.success("⚡ Automatic split DISABLED: Faster solving with pure strategies only.")
        force_split_enabled = False

    # Package Parameters
    be_params = {
        'R_W': p['R_W'], 'R_W_ideal': p['R_W_ideal'], 'R_M': p['R_M'], 'R_M_ideal': p['R_M_ideal'],
        'f_W': p['f_W'], 'c_W_empty': p['c_W_empty'], 'C_shift': p['C_shift'], 'S_max': p['S_max'],
        'F_freq': p['F_freq'], 'D_week': p['D_week'], 'p_bias': p['p_bias'],
        'beta': {'rest': p['beta_rest'], 'cardboard': p['beta_card'], 'glass': p['beta_glass']},
        'C_W': {'rest': p['cw_rest'], 'cardboard': p['cw_card'], 'glass': p['cw_glass']},
        'C_M': {'rest': p['cm_rest'], 'cardboard': p['cm_card'], 'glass': p['cm_glass']},
        'stop_cost_dump': p['stop_cost_dump'],
        'V_truck': {'rest': p['v_truck_rest'], 'cardboard': p['v_truck_card'], 'glass': p['v_truck_glass']},
        'mobile_only_unreachable': mobile_only_unreachable,
        'split_strategy_enabled': split_strategy_enabled,
        'force_split_enabled': force_split_enabled,
        'underground_containers': st.session_state['underground_containers']
    }

    st.markdown("---")
    
    # --- RUN BUTTON LOGIC ---
    if st.button("Run Optimization", type="primary"):
        required_files = [data_files["demand"], data_files["water"], data_files["mobile"]]
        if not all(required_files):
            st.error("❌ Missing primary data files (Demand/Water OD/Mobile OD)")
        else:
            with st.spinner("Running Optimization..."):
                # Reset old results
                st.session_state['diag_ran'] = False
                st.session_state['diag_results'] = None
                
                try:
                    time_limit_val = solver_time_limit if solver_time_limit > 0 else None
                    
                    if use_augmecon:
                        # AUGMECON2 method with automatic grid and acceleration
                        fig, res_df, details, allocations, dist_distributions = run_augmecon(
                            be_params, data_files, n_grid_points, solver_name, st.progress(0),
                            time_limit=time_limit_val, mip_gap=solver_mip_gap,
                            delta=augmecon_delta, early_exit=augmecon_early_exit
                        )
                    else:
                        # Standard epsilon-constraint method
                        fig, res_df, details, allocations, dist_distributions = run_optimization(
                            be_params, data_files, epsilons, solver_name, st.progress(0), 
                            time_limit=time_limit_val, mip_gap=solver_mip_gap
                        )
                    
                    st.session_state['opt_results'] = res_df
                    st.session_state['opt_details'] = details
                    st.session_state['opt_allocations'] = allocations # Store Allocations
                    st.session_state['opt_distance_dist'] = dist_distributions  # Store Distance Distributions
                    st.session_state['opt_fig'] = fig
                    
                    if res_df is not None and not res_df.empty:
                        # Check if we have ANY feasible results
                        feasible_count = res_df['Cost'].notna().sum()
                        if feasible_count > 0:
                            st.session_state['status'] = "success"
                        else:
                            st.session_state['status'] = "infeasible"
                    else:
                        st.session_state['status'] = "infeasible"
                        
                except Exception as e:
                    st.session_state['status'] = "error"
                    st.session_state['error_msg'] = str(e)

# --- RESULTS DISPLAY ---

# Helper function to format epsilon values nicely
def format_epsilon(eps):
    """Format epsilon with max 2 decimal places."""
    if eps >= 1000:
        return f"{eps:.0f}"  # No decimals for large numbers
    elif eps >= 1:
        return f"{eps:.2f}"  # Max 2 decimals
    else:
        return f"{eps:.2f}"

if st.session_state['status'] == "success":
    st.success("✅ Optimization Complete")
    df = st.session_state['opt_results']
    details = st.session_state['opt_details']
    allocations = st.session_state['opt_allocations']
    dist_distributions = st.session_state['opt_distance_dist']
    
    # Filter to only feasible results for plotting, but show all in table
    df_all = df.copy()
    df = df[df['Cost'].notna()].copy()
    
    # Show warning if some were infeasible
    infeasible_count = df_all['Cost'].isna().sum()
    if infeasible_count > 0:
        st.warning(f"⚠️ {infeasible_count} epsilon value(s) were infeasible (ε too small). Showing {len(df)} feasible results.")
    
    # --- FIX: Sort results by Cost for clean gradient plotting ---
    if not df.empty:
        df = df.sort_values(by="Cost")
    
    c1, c2 = st.columns(2)
    with c1:
        # Graph 1: Cost vs Total Penalty (Biased)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Cost'], df['Total_Penalty'], marker='o', linestyle='--')
        
        # Smart label placement to avoid overlap
        if ADJUST_TEXT_AVAILABLE:
            texts = []
            for i, row in df.iterrows():
                texts.append(ax.text(row['Cost'], row['Total_Penalty'], f"ε={format_epsilon(row['Epsilon'])}", fontsize=9))
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        else:
            # Fallback: annotate every 3rd point to reduce clutter
            for idx, (i, row) in enumerate(df.iterrows()):
                if idx % 3 == 0 or idx == len(df) - 1:  # Show first, last, and every 3rd
                    ax.text(row['Cost'], row['Total_Penalty'], f" ε={format_epsilon(row['Epsilon'])}", fontsize=9)
        
        ax.set_xlabel("Cost (€)")
        ax.set_ylabel("Penalty (Biased for Solving)")
        ax.set_title("Optimization Trade-off")
        ax.grid(True)
        st.pyplot(fig)

    with c2:
        # Graph 2: Cost vs Actual Walking Distance (Physical Reality)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if 'Avg_Distance' in df.columns:
            ax2.plot(df['Cost'], df['Avg_Distance'], marker='o', linestyle='--', color='green')
            
            # Smart label placement to avoid overlap
            if ADJUST_TEXT_AVAILABLE:
                texts = []
                for i, row in df.iterrows():
                    texts.append(ax2.text(row['Cost'], row['Avg_Distance'], f"ε={format_epsilon(row['Epsilon'])}", fontsize=9))
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
            else:
                # Fallback: annotate every 3rd point to reduce clutter
                for idx, (i, row) in enumerate(df.iterrows()):
                    if idx % 3 == 0 or idx == len(df) - 1:  # Show first, last, and every 3rd
                        ax2.text(row['Cost'], row['Avg_Distance'], f" ε={format_epsilon(row['Epsilon'])}", fontsize=9)
            
            ax2.set_xlabel("Cost (€)")
            ax2.set_ylabel("Avg Walking Distance (m)")
            ax2.set_title("Physical Trade-off (Avg Distance)")
            ax2.grid(True)
            st.pyplot(fig2)
        else:
            st.warning("Average distance data unavailable in this backend version.")
            
    # Summary Tables and Stats
    c_sum1, c_sum2 = st.columns([2, 1])
    with c_sum1:
        st.write("### Scenario Summary")
        st.dataframe(df, width='stretch')
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "summary.csv", "text/csv")
    
    with c_sum2:
        st.write("### Key Statistics")
        if 'Avg_Distance' in df.columns:
            stats = df[['Cost', 'Avg_Distance', 'Max_Distance']].describe().loc[['mean', '50%', 'min', 'max']]
            st.dataframe(stats, width='stretch')
        
        # Solver performance summary
        if 'Solve_Time_s' in df.columns:
            st.markdown("---")
            st.write("### Solver Performance")
            total_time = df['Solve_Time_s'].sum()
            avg_time = df['Solve_Time_s'].mean()
            max_time = df['Solve_Time_s'].max()
            st.metric("Total Time", f"{total_time:.2f}s")
            c_t1, c_t2 = st.columns(2)
            c_t1.metric("Avg Time/ε", f"{avg_time:.2f}s")
            c_t2.metric("Max Time/ε", f"{max_time:.2f}s")
            
            if 'Gap' in df.columns and df['Gap'].notna().any():
                avg_gap = df['Gap'].mean()
                st.metric("Avg Gap", f"{avg_gap*100:.2f}%")

    # --- WALKING DISTANCE HISTOGRAM SECTION ---
    st.markdown("---")
    st.subheader("🚶 Walking Distance Distribution")
    
    available_epsilons_hist = list(dist_distributions.keys()) if dist_distributions else []
    if available_epsilons_hist:
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            selected_epsilon_hist = st.selectbox("Select Epsilon Scenario for Histogram", available_epsilons_hist, key="epsilon_hist")
        
        with col_ctrl2:
            st.write(" **Axis Settings**")
            c_ax1, c_ax2, c_ax3 = st.columns(3)
            use_manual_axes = c_ax1.checkbox("Manual Axis Limits", value=False)
            if use_manual_axes:
                max_x_dist = c_ax2.number_input("Max Distance (x-axis)", min_value=10, value=300, step=10)
                max_y_count = c_ax3.number_input("Max Households (y-axis)", min_value=1, value=1000, step=50)
            else:
                max_x_dist = None
                max_y_count = None

        dist_data = dist_distributions[selected_epsilon_hist]
        
        if dist_data and (dist_data['water_distances'] or dist_data['mobile_distances'] or dist_data.get('underground_distances', [])):
            
            # Create bins of 10 meters
            # Create bins of 10 meters
            if max_x_dist:
                limit_dist = max_x_dist
            else:
                limit_dist = max(
                    max(dist_data['water_distances']) if dist_data['water_distances'] else 0,
                    max(dist_data['mobile_distances']) if dist_data['mobile_distances'] else 0,
                    max(dist_data.get('underground_distances', [0])) if dist_data.get('underground_distances', []) else 0
                )
            
            bins = list(range(0, int(limit_dist) + 20, 10))
            if max_x_dist:
                # Add one extra bin just to catch overflow if any (though we want to cut it off visually)
                 # Actually, for histogram, "cutting off" x-axis usually means ignoring values > max.
                 pass

            bin_labels = [f"{b}-{b+10}" for b in bins[:-1]]
            
            # Count households in each bin for water
            water_counts = [0] * (len(bins) - 1)
            for d in dist_data['water_distances']:
                if max_x_dist and d > max_x_dist: continue
                bin_idx = int(d // 10)
                if bin_idx < len(water_counts):
                    water_counts[bin_idx] += 1
            
            # Count households in each bin for mobile
            mobile_counts = [0] * (len(bins) - 1)
            for d in dist_data['mobile_distances']:
                if max_x_dist and d > max_x_dist: continue
                bin_idx = int(d // 10)
                if bin_idx < len(mobile_counts):
                    mobile_counts[bin_idx] += 1
            
            # Count households in each bin for underground
            underground_counts = [0] * (len(bins) - 1)
            for d in dist_data.get('underground_distances', []):
                if max_x_dist and d > max_x_dist: continue
                bin_idx = int(d // 10)
                if bin_idx < len(underground_counts):
                    underground_counts[bin_idx] += 1
            
            # Create summary dataframe
            summary_df = pd.DataFrame({
                'Distance (m)': bin_labels,
                'Water (households)': water_counts,
                'Mobile (households)': mobile_counts,
                'Underground (households)': underground_counts,
                'Total (households)': [w + m + u for w, m, u in zip(water_counts, mobile_counts, underground_counts)]
            })
            # Filter out empty rows unless we want to show the full range
            if not max_x_dist:
                summary_df = summary_df[(summary_df['Water (households)'] > 0) | (summary_df['Mobile (households)'] > 0) | (summary_df['Underground (households)'] > 0)]
            
            c_hist1, c_hist2 = st.columns(2)
            
            with c_hist1:
                # Bar chart
                fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                x = range(len(summary_df))
                width = 0.25
                ax_hist.bar([i - width for i in x], summary_df['Water (households)'], width, label='Water', color='blue', alpha=0.7)
                ax_hist.bar([i for i in x], summary_df['Mobile (households)'], width, label='Mobile', color='red', alpha=0.7)
                ax_hist.bar([i + width for i in x], summary_df['Underground (households)'], width, label='Underground', color='green', alpha=0.7)
                
                # Apply Y-Axis Limit if set
                if max_y_count:
                    ax_hist.set_ylim(0, max_y_count)

                ax_hist.set_xlabel('Distance (metres)')
                ax_hist.set_ylabel('Number of Households')
                ax_hist.set_title(f'Household Walking Distance Distribution (ε={selected_epsilon_hist})')
                ax_hist.set_xticks(x)
                ax_hist.set_xticklabels(summary_df['Distance (m)'], rotation=45, ha='right')
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_hist)
            
            with c_hist2:
                st.write("**Distance Summary Table**")
                st.dataframe(summary_df, width='stretch', hide_index=True)
                
                # Quick stats
                total_water = sum(water_counts)
                total_mobile = sum(mobile_counts)
                total_underground = sum(underground_counts)
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("Water", total_water)
                c_m2.metric("Mobile", total_mobile)
                c_m3.metric("Underground", total_underground)
                
                # Download button
                csv_hist = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Distance Summary", csv_hist, f"distance_distribution_{selected_epsilon_hist}.csv", "text/csv")
            
            # --- HOUSEHOLD DISTANCE MAP ---
            st.markdown("---")
            st.write("### 🗺️ Household Walking Distance Map")
            
            # Get allocation data for this epsilon
            if selected_epsilon_hist in allocations:
                alloc_for_map = allocations[selected_epsilon_hist]
                
                # Check if we have household coordinates
                if not alloc_for_map.empty and 'hh_lat' in alloc_for_map.columns and 'hh_lon' in alloc_for_map.columns:
                    # Get unique households with their max distance
                    hh_map_data = alloc_for_map.groupby('Household').agg({
                        'Distance': 'max',
                        'Type': 'first',
                        'hh_lat': 'first',
                        'hh_lon': 'first'
                    }).reset_index()
                    hh_map_data = hh_map_data.dropna(subset=['hh_lat', 'hh_lon'])
                    
                    if not hh_map_data.empty and FOLIUM_AVAILABLE:
                        # Create color scale based on distance
                        max_distance = hh_map_data['Distance'].max()
                        min_distance = hh_map_data['Distance'].min()
                        
                        def get_color(distance):
                            """Return color from green (close) to red (far)"""
                            if max_distance == min_distance:
                                return '#00FF00'
                            # Normalize to 0-1
                            norm = (distance - min_distance) / (max_distance - min_distance)
                            # Green (0,255,0) to Yellow (255,255,0) to Red (255,0,0)
                            if norm < 0.5:
                                r = int(255 * (norm * 2))
                                g = 255
                            else:
                                r = 255
                                g = int(255 * (1 - (norm - 0.5) * 2))
                            return f'#{r:02x}{g:02x}00'
                        
                        # Calculate map center
                        center_lat = hh_map_data['hh_lat'].mean()
                        center_lon = hh_map_data['hh_lon'].mean()
                        
                        # Create Folium map
                        m_hh = folium.Map(location=[center_lat, center_lon], zoom_start=14)
                        
                        # Add circle markers for each household
                        for idx, row in hh_map_data.iterrows():
                            color = get_color(row['Distance'])
                            
                            popup_html = f"""
                            <div style="font-family: Arial; min-width: 120px;">
                                <b>Household:</b> {row['Household']}<br>
                                <b>Distance:</b> {row['Distance']:.1f} m<br>
                                <b>Assigned to:</b> {row['Type']}
                            </div>
                            """
                            
                            folium.CircleMarker(
                                location=[row['hh_lat'], row['hh_lon']],
                                radius=3,
                                color=color,
                                fill=True,
                                fillColor=color,
                                fillOpacity=0.7,
                                popup=folium.Popup(popup_html, max_width=200),
                                tooltip=f"{row['Distance']:.0f}m"
                            ).add_to(m_hh)
                        
                        # Render map
                        st_folium(m_hh, width=700, height=450, key=f"hh_map_{selected_epsilon_hist}")
                        
                        # Add legend
                        st.markdown(f"""
                        **Legend:** 🟢 Short distance ({min_distance:.0f}m) → 🟡 Medium → 🔴 Long distance ({max_distance:.0f}m)
                        
                        *Each dot represents a household. Color indicates walking distance to assigned facility.*
                        """)
                    else:
                        if not FOLIUM_AVAILABLE:
                            st.warning("📦 Install `streamlit-folium` for interactive maps: `pip install folium streamlit-folium`")
                        else:
                            st.info("No household coordinate data available for map.")
                else:
                    st.info("Upload household coordinate files to see the distance map.")
        else:
            st.info("No distance distribution data available for this scenario.")
    else:
        st.info("Run optimization to see distance distributions.")

    st.markdown("---")
    st.subheader("📍 Detailed Facility Plan & Map")
    
    available_epsilons = list(details.keys())
    if available_epsilons:
        selected_epsilon = st.selectbox("Select Epsilon Scenario", available_epsilons)
        facility_df = details[selected_epsilon]
        alloc_df = allocations[selected_epsilon]
        
        if not facility_df.empty:
            c_map, c_list = st.columns([3, 2])
            
            # 1. INTERACTIVE MAP (Folium)
            with c_map:
                # Check for valid coordinates
                if 'lat' in facility_df.columns and 'lon' in facility_df.columns and facility_df['lat'].notnull().any():
                    map_df = facility_df.copy().dropna(subset=['lat', 'lon'])
                    
                    if FOLIUM_AVAILABLE and not map_df.empty:
                        # Calculate map center
                        center_lat = map_df['lat'].mean()
                        center_lon = map_df['lon'].mean()
                        
                        # Create Folium map
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
                        
                        # Add markers for each facility
                        for idx, row in map_df.iterrows():
                            # Color based on type
                            if row['Type'] == 'Water':
                                color = 'blue'
                                icon_type = 'tint'  # water droplet
                            elif row['Type'] == 'Underground':
                                color = 'green'
                                icon_type = 'arrow-down'  # underground
                            else:
                                color = 'red'
                                icon_type = 'truck'  # mobile/truck
                            
                            # Check if this facility is selected
                            is_selected = (st.session_state['selected_facility'] == row['ID'])
                            
                            # Get color for popup
                            type_colors = {'Water': '#0066cc', 'Mobile': '#cc0000', 'Underground': '#006600'}
                            
                            # Create popup content
                            popup_html = f"""
                            <div style="font-family: Arial; min-width: 150px;">
                                <h4 style="margin: 0; color: {type_colors.get(row['Type'], '#333')};">
                                    {row['ID']}
                                </h4>
                                <hr style="margin: 5px 0;">
                                <b>Type:</b> {row['Type']}<br>
                                <b>Clients:</b> {row['Clients']}<br>
                                <b>Demand:</b> {row['Allocated_Demand']:.1f} L<br>
                                <b>Capacity Units:</b> {row['Capacity_Units']:.0f}
                            </div>
                            """
                            
                            # Add marker with larger size if selected
                            folium.Marker(
                                location=[row['lat'], row['lon']],
                                popup=folium.Popup(popup_html, max_width=250),
                                tooltip=f"Click: {row['ID']}",
                                icon=folium.Icon(
                                    color='green' if is_selected else color,
                                    icon=icon_type,
                                    prefix='fa'
                                )
                            ).add_to(m)
                            
                            # Add circle highlight if selected
                            if is_selected:
                                folium.Circle(
                                    location=[row['lat'], row['lon']],
                                    radius=30,
                                    color='green',
                                    fill=True,
                                    fillOpacity=0.3
                                ).add_to(m)
                        
                        # Render map and capture click events
                        map_data = st_folium(m, width=500, height=400, key=f"map_{selected_epsilon}")
                        
                        # Handle click events
                        if map_data and map_data.get('last_object_clicked'):
                            clicked_lat = map_data['last_object_clicked'].get('lat')
                            clicked_lon = map_data['last_object_clicked'].get('lng')
                            
                            # Find which facility was clicked (nearest to click point)
                            if clicked_lat and clicked_lon:
                                distances = ((map_df['lat'] - clicked_lat)**2 + (map_df['lon'] - clicked_lon)**2)
                                nearest_idx = distances.idxmin()
                                clicked_facility = map_df.loc[nearest_idx, 'ID']
                                
                                if clicked_facility != st.session_state['selected_facility']:
                                    st.session_state['selected_facility'] = clicked_facility
                                    st.rerun()
                        
                        st.caption("🔵 Water | 🔴 Mobile | 🟢 Underground | ⭐ Selected — Click a marker for details")
                        
                    else:
                        # Fallback to basic st.map if Folium not available
                        if not FOLIUM_AVAILABLE:
                            st.warning("📦 Install `streamlit-folium` for interactive maps: `pip install folium streamlit-folium`")
                        type_colors = {'Water': '#0000FF', 'Mobile': '#FF0000', 'Underground': '#00FF00'}
                        map_df['color'] = map_df['Type'].apply(lambda x: type_colors.get(x, '#888888'))
                        map_df['size'] = map_bubble_size 
                        st.map(map_df, color='color', size='size')
                        st.caption("🔵 Water | 🔴 Mobile | 🟢 Underground")

                    # Simple dot-only map (always available) using Streamlit's base map
                    if not map_df.empty:
                        # Show Water (blue), Mobile (red), and Underground (green) as plain circles
                        dots_df = map_df[['lat', 'lon', 'Type']].copy()
                        dots_df['color'] = dots_df['Type'].apply(lambda t: '#0000FF' if t == 'Water' else ('#00FF00' if t == 'Underground' else '#FF0000'))
                        dots_df['size'] = 8
                        st.markdown("**Basic Map (dots only)**")
                        st.map(dots_df, color='color', size='size')
                        st.caption("🔵 Water | 🔴 Mobile | 🟢 Underground — basic circles without custom icons (always shown)")
                else:
                    st.warning("⚠️ No valid map data found. Upload coordinate files.")
            
            # 2. FACILITY LIST + SELECTED DETAILS
            with c_list:
                st.write(f"**Open Facilities (Epsilon={selected_epsilon})**")
                
                # Show selected facility details if any
                if st.session_state['selected_facility']:
                    selected_row = facility_df[facility_df['ID'] == st.session_state['selected_facility']]
                    if not selected_row.empty:
                        row = selected_row.iloc[0]
                        st.success(f"**Selected: {row['ID']}**")
                        col1, col2 = st.columns(2)
                        col1.metric("Type", row['Type'])
                        col2.metric("Clients", int(row['Clients']))
                        col1.metric("Demand (L)", f"{row['Allocated_Demand']:.1f}")
                        col2.metric("Capacity", f"{row['Capacity_Units']:.0f}")
                        
                        # Show households assigned to this facility
                        facility_allocs = alloc_df[alloc_df['Facility'] == row['ID']]
                        if not facility_allocs.empty:
                            with st.expander(f"📋 {len(facility_allocs)} Households assigned"):
                                st.dataframe(facility_allocs[['Household', 'Strategy', 'Distance']], 
                                           width='stretch', height=150)
                        
                        if st.button("Clear Selection"):
                            st.session_state['selected_facility'] = None
                            st.rerun()
                        st.markdown("---")
                
                # Facility table
                cols = ['ID', 'Type', 'Capacity_Units', 'Clients', 'Allocated_Demand']
                st.dataframe(facility_df[cols], width='stretch')
                
                csv_fac = facility_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Facility List", csv_fac, f"facilities_{selected_epsilon}.csv", "text/csv")
        
        # 3. HOUSEHOLD ALLOCATIONS EXPANDER
        with st.expander("Show Household Allocations"):
            st.info("This table shows exactly which household is assigned to which facility.")
            st.dataframe(alloc_df, width='stretch')
            csv_alloc = alloc_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Allocations", csv_alloc, f"allocations_{selected_epsilon}.csv", "text/csv")

elif st.session_state['status'] == "infeasible":
    st.error("❌ Model Infeasible: No valid solution found.")
    st.warning("""
**Common causes:**
1. **ID mismatch**: `fid` in demand file doesn't match `origin_id` in OD matrices
2. **Distance too small**: `R_W` or `R_M` excludes all households
3. **Empty data**: No valid pairs after filtering

👇 **Run Diagnostics below** to identify the exact issue.
    """)

elif st.session_state['status'] == "error":
    st.error("⚠️ Optimization Error")
    st.error(st.session_state['error_msg'])

elif st.session_state['status'] == "input":
    st.info("👈 Upload your data files and click 'Run Optimization' to start.")

# --- ALWAYS VISIBLE DIAGNOSTICS (Bottom of Page) ---
st.markdown("---")
with st.expander("🛠️ Troubleshooting & Diagnostics (Run anytime)", expanded=(st.session_state['status'] == "infeasible")):
    st.info("Use this tool to verify constraints, even if the model found a solution.")
    
    if st.button("Run Diagnostics Tool", type="secondary"):
        with st.spinner("Diagnosing..."):
            try:
                # Diagnostics handles coord files too if passed
                err, res = run_diagnostics(be_params, data_files, solver_name)
                st.session_state['diag_results'] = (err, res)
                st.session_state['diag_ran'] = True
            except Exception as e:
                st.error(f"Diagnostics Failed: {e}")

    if st.session_state['diag_ran']:
        err, res = st.session_state['diag_results']
        if err: st.error(err)
        else:
            rep, unreach = res
            st.markdown("#### Diagnostic Report")
            c1, c2, c3 = st.columns(3)
            c1.metric("Unreachable", rep["Totally Unreachable (Distance)"])
            c2.metric("Unserved", rep["Unserved Households (Optimization)"])
            c3.metric("Fleet Shortage", f"{rep['Missing Shifts (Fleet Constraint)']:.1f} Shifts")
            
            c4, c5 = st.columns(2)
            c4.metric("Mob Overflow", f"{rep['Mobile Capacity Overflow (L)']:.0f} L")
            c5.metric("Water Overflow", f"{rep['Water Capacity Overflow (L)']:.0f} L")
            
            if rep["Totally Unreachable (Distance)"] > 0:
                st.warning(f"**Distance:** {len(unreach)} households too far.")
            if rep["Missing Shifts (Fleet Constraint)"] > 0.1:
                st.warning("**Fleet:** Increase `Stops/Shift` or `Visits/Day`.")