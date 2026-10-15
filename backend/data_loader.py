# backend/data_loader.py
"""
Data loading and preprocessing. 

Functions:
    - extract_coords: Pull lat/lon from a dataframe
    - load_coords_from_file: Read uploaded coordinate CSV
    - clean_dict_keys: Normalize float IDs to int
    - load_data: Master loader that returns all structures needed by the model
    - compute_network_distances: Calculate walking distances using OSMnx
"""

import pandas as pd
from pyproj import Transformer

from .parameters import WASTE_TYPES


def _read_csv_smart(file_or_path, **kwargs):
    """
    Read a CSV with auto-detect of comma/semicolon separator.
    Accepts either a path/str or a file-like object (Streamlit uploads).
    Handles a leading UTF-8 BOM (common from Dutch Excel exports).
    """
    is_file_obj = hasattr(file_or_path, "seek")
    read_kwargs = {"encoding": "utf-8-sig", **kwargs}
    df = pd.read_csv(file_or_path, **read_kwargs)
    if len(df.columns) <= 1:
        if is_file_obj:
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path, sep=";", **read_kwargs)
    return df

# RD New (EPSG:28992) → WGS84 (EPSG:4326) transformer
# Created once at module level for efficiency
_RD_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# Try to import OSMnx for network distance calculation. This is only used for the underground containers.
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
#Print warning if osmnx is not installed > Not needed if using application v2 as it has osmnx preinstalled in installer
except ImportError:
    OSMNX_AVAILABLE = False
    print("[WARNING] osmnx not installed. Underground containers will use Euclidean distance.")
    print("[WARNING] Install with: pip install osmnx")


def rd_to_wgs84(x, y):
    """
    Convert Dutch RD-New (EPSG:28992) coordinates to WGS84 (EPSG:4326) using pyproj.
    
    Args:
        x: RD X coordinate (easting) > CAUTION: Check if x and y are swapped in your data.
        y: RD Y coordinate (northing)
        
    Returns:
        tuple: (latitude, longitude) in WGS84
    """
    lon, lat = _RD_TO_WGS84.transform(x, y)
    return (lat, lon)


def compute_network_distances_osmnx(container_coords, household_coords, max_distance=300):
    """
    Compute network walking distances from underground containers to households using OSMnx. 
    These coords come from the ui (either via the map or manual input) and are in WGS84 (lat/lon)
    
    Args:
        container_coords: list of (id, lat, lon) tuples for underground containers
        household_coords: list of (id, lat, lon) tuples for households
        max_distance: maximum distance in meters to consider (for filtering)
        
    Returns:
        dict: {(household_id, container_id): network_distance_meters}
    """
    if not OSMNX_AVAILABLE or not container_coords or not household_coords:
        return {}
    
    try:
        # Filter households to only those potentially within range of containers
        # This makes sure that the network is not unnecessarily large and takes too much time to load
        # Use Euclidean pre-filter to avoid downloading huge networks > using road network would take too much time
        # 300m ~ 0.0027 lat degrees, 0.0045 lon degrees at Amsterdam latitude
        filtered_households = []
        for h_id, h_lat, h_lon in household_coords:
            for c_id, c_lat, c_lon in container_coords:
                # Quick Euclidean check (~300m) > 111000 and 67000 are conversion factors for lat/lon to meters at Amsterdam latitude
                lat_diff = abs(h_lat - c_lat) * 111000  # meters
                lon_diff = abs(h_lon - c_lon) * 67000   # meters at ~52° lat
                euclidean_dist = (lat_diff**2 + lon_diff**2)**0.5
                if euclidean_dist <= max_distance:
                    filtered_households.append((h_id, h_lat, h_lon))
                    break #Used to stop checking other containers > if household is within 300m of one container, it is not necessary to check other containers
        
        #prevents downloading the network if no households are within the range of a UC        
        if not filtered_households:
            return {}
        
        # Download network for each container
        distances = {}
        
        for c_id, c_lat, c_lon in container_coords:
            try:
                # Download walking network around this container (buffer = max_distance + 50m)
                G = ox.graph_from_point((c_lat, c_lon), dist=max_distance + 50, network_type='walk')
                
                # Find nodes BEFORE projecting (using lat/lon)
                # This is necessary if the lon lat are not exactly on a road > which they usually are not
                # These entry costs are not included in the total walking distance
                c_node = ox.nearest_nodes(G, c_lon, c_lat)
                
                # Map household nodes before projection
                household_node_map = {}
                for h_id, h_lat, h_lon in filtered_households:
                    # Check if this household is near this container
                    lat_diff = abs(h_lat - c_lat) * 111000
                    lon_diff = abs(h_lon - c_lon) * 67000
                    euclidean_dist = (lat_diff**2 + lon_diff**2)**0.5
                    # Searches nearest node on network for households that are within 300m of a container and adds them to the map
                    if euclidean_dist <= max_distance:
                        try:
                            h_node = ox.nearest_nodes(G, h_lon, h_lat)
                            household_node_map[h_id] = h_node
                        except:
                            pass
                
                # Now project and compute distances in meters
                G_proj = ox.project_graph(G)
                
                for h_id, h_node in household_node_map.items():
                    try:
                        length = nx.shortest_path_length(G_proj, c_node, h_node, weight='length')
                        # Create dict of distances for in-range households
                        if length <= max_distance:
                            distances[(h_id, c_id)] = length
                    except:
                        pass
            # Failsafe        
            except Exception as e:
                continue
        
        return distances
    # Failsafe    
    except Exception as e:
        return {}



# Type fixer for IDs
def clean_id(val):
    """
    Helper to ensure IDs are consistent (int if possible, otherwise string).
    Converts 1.0 -> 1.
    """
    try:
        if isinstance(val, float) and val.is_integer():
            return int(val)
    except:
        pass
    return val


def extract_coords(df, id_col):
    """
    Scans dataframe for coords and maps them to an ID.
    Automatically detects and converts RD coordinates to WGS84.
    
    Supported Columns (Case-Insensitive):
    - Latitude: 'lat', 'latitude'
    - Longitude: 'lon', 'longitude'
    - RD coordinates: 'x', 'y'
    
    Args:
        df (pd.DataFrame): The dataframe to search.
        id_col (str): The name of the ID column.
        
    Returns:
        dict: Mapping of {id: (lat, lon)} in WGS84
    """
    # Check for explicit lat/lon columns first > use next to find first column that matches
    lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude']), None)
    lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude']), None)
    
    # If no lat/lon, check for X/Y
    x_col = next((c for c in df.columns if c.lower() == 'x'), None)
    y_col = next((c for c in df.columns if c.lower() == 'y'), None)
    
    coords = {}
    
    # Checks if lat/lon columns exist otherwise checks for x/y columns
    if lat_col and lon_col:
        # Explicit lat/lon columns - use directly
        for _, row in df.iterrows():
            coords[row[id_col]] = (row[lat_col], row[lon_col])
    elif x_col and y_col:
        # X/Y columns - check if RD or WGS84
        sample_x = df[x_col].iloc[0] if len(df) > 0 else 0
        sample_y = df[y_col].iloc[0] if len(df) > 0 else 0
        
        # RD coordinates for Amsterdam are around the following values:
        # X: ~100,000 - 200,000
        # Y: ~470,000 - 500,000
        is_rd = (50000 < sample_x < 300000) and (300000 < sample_y < 700000)
        
        # If they are rd-new then convert them to wgs84
        if is_rd:
            for _, row in df.iterrows():
                try:
                    lat, lon = rd_to_wgs84(row[x_col], row[y_col])
                    coords[row[id_col]] = (lat, lon)
                except:
                    pass
        else:
            # Assume X=lon, Y=lat
            for _, row in df.iterrows():
                coords[row[id_col]] = (row[y_col], row[x_col])
            
    return coords


def load_coords_from_file(file_obj, id_col_name='fid', prefix=''):
    """
    Reads a specific uploaded coordinate file.
    Handles CSV that are in dutch or english format (semicolon vs comma) and ID formatting.
    Is made for QGIS exports, but can handle other formats if similar.
    Also can differentiate between rd-new and wgs84. CANNOT HANDLE OTHER FORMATS > WILL ASSUME WGS84 AND BREAK.
    
    Args:
        file_obj: Streamlit file object.
        id_col_name (str): Name of ID column to find.
        prefix (str): Prefix to add to the ID (e.g., 'W-' or 'M-').
        
    Returns:
        dict: Mapping of {prefix+id: (lat, lon)}
    """
    coords = {}
    if file_obj is not None:
        try:
            # 1. LOADING (Handle ; separator)§
            try:
                # Read first with standard seperator ,
                df = pd.read_csv(file_obj)
                if len(df.columns) <= 1: 
                    # Reset pointer to start of file again
                    file_obj.seek(0)
                    # Read with ; seperator (dutch excel)
                    df = pd.read_csv(file_obj, sep=';')
            # If first read fails for some reason, this tries to fix it with the ; sep
            except:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, sep=';')

            df.columns = [c.lower() for c in df.columns]
            
            # 2. ID FINDING > checks default (fid) first, if fail then id
            id_col = next((c for c in df.columns if id_col_name.lower() in c.lower()), None)
            if not id_col: 
                id_col = next((c for c in df.columns if 'id' in c.lower()), None)
            
            # 3. X/Y FINDING > checks default x and y first, if fail then lat/lon
            x_col = next((c for c in df.columns if c.lower() == 'x'), None)
            y_col = next((c for c in df.columns if c.lower() == 'y'), None)
            
            if id_col and x_col and y_col:
                # Check if coordinates are RD or WGS84 > same as function above
                sample_x = df[x_col].iloc[0] if len(df) > 0 else 0
                sample_y = df[y_col].iloc[0] if len(df) > 0 else 0
                
                # RD coordinates: X ~ 100,000-200,000, Y ~ 450,000-500,000
                # WGS84 for Amsterdam: X(lon) ~ 4.8-5.0, Y(lat) ~ 52.3-52.4
                is_rd = (50000 < sample_x < 300000) and (300000 < sample_y < 700000)
                
                if is_rd:
                    for _, row in df.iterrows():
                        # Set id as integer
                        raw_id = row[id_col]
                        try:
                            if isinstance(raw_id, float) and raw_id.is_integer():
                                raw_id = int(raw_id)
                        except: 
                            pass
                        # change id to string and add a prefix > W for Waterborne and M for Mobile
                        clean_id = f"{prefix}{raw_id}" if prefix else raw_id
                        try:
                            lat, lon = rd_to_wgs84(row[x_col], row[y_col])
                            coords[clean_id] = (lat, lon)
                        except:
                            pass
                # Assumed that format is WSG84
                else:
                    for _, row in df.iterrows():
                        raw_id = row[id_col]
                        try:
                            if isinstance(raw_id, float) and raw_id.is_integer():
                                raw_id = int(raw_id)
                        except: 
                            pass
                        
                        clean_id = f"{prefix}{raw_id}" if prefix else raw_id
                        # X = longitude, Y = latitude -> store as (lat, lon)
                        coords[clean_id] = (row[y_col], row[x_col])
                
            else:
                pass
                
        except Exception as e:
            pass
            
    return coords

# Not sure if still relevant, but a failsafe for the demand file
def clean_dict_keys(d):
    """
    Ensures dictionary keys (IDs) are integers if they look like floats.
    Prevents mismatch between '1.0' and '1'.
    """
    new_d = {}
    for (i, dest), val in d.items():
        try:
            if isinstance(i, float) and i.is_integer(): 
                i = int(i)
        except: 
            pass
        new_d[(i, dest)] = val
    return new_d


########### MAIN DATA LOADER ###########

def load_data(demand_file, water_od_file, mobile_od_file, u_od_files, coord_files, params):
    """
    Loads all CSV inputs, cleans IDs, calculates penalties, and merges coordinates.

    Args:
        demand_file: path or file-obj with columns fid, demand_rest, demand_cardboard,
            demand_glass (and optionally X, Y for coordinates).
        water_od_file: path or file-obj with columns origin_id, destination_id, network_cost.
        mobile_od_file: same schema.
        u_od_files: dict mapping waste fraction -> OD file. Keys must be a subset of
            {'rest', 'cardboard', 'glass'}. Any missing fraction yields empty sets for
            that fraction (the model can still solve, just without that UC option).
        coord_files: optional dict with keys 'demand', 'water', 'mobile' pointing at
            separate coordinate CSVs (used only for map rendering).
        params: parameter dict (see backend.parameters.DEFAULT_PARAMS).

    Returns:
        tuple: (data_tuple, error_string)
            data_tuple contains:
                I, W, M, feas_w, feas_m, demand, pen_w, pen_m, pen_m_raw,
                coord_map, dist_water, dist_mobile
            error_string is None on success, else an error message.
    """
    print("--- Loading Data ---")
    try:
        # Load demand file with households and businesses data
        demand_df = _read_csv_smart(demand_file)

        if 'fid' not in demand_df.columns:
             raise ValueError(f"Missing 'fid' column. Found: {demand_df.columns}")

        # sets demand at zero if missing
        for col in ['demand_rest', 'demand_cardboard', 'demand_glass']:
            if col not in demand_df.columns:
                demand_df[col] = 0.0

        # Demand in the source CSV is kg/week per household. The model's capacities
        # are also in kg (C_W, C_M, C_U, V_truck) — no conversion needed.

        # create demand data dictionary
        demand_data = {}
        for _, row in demand_df.iterrows():
            i = row['fid']
            try:
                if isinstance(i, float) and i.is_integer():
                    i = int(i)
            except:
                pass

            demand_data[(i, 'rest')]      = row['demand_rest']
            demand_data[(i, 'cardboard')] = row['demand_cardboard']
            demand_data[(i, 'glass')]     = row['demand_glass']

        # Creates list of all household and Business IDs and dictionary of coordinates
        I = list(set(k[0] for k in demand_data.keys()))
        coord_map = extract_coords(demand_df, 'fid')

        # Load water OD matrix
        w_df = _read_csv_smart(water_od_file)
        
        # Filters out points that are too far away based on maximum walking distance
        w_df = w_df[w_df['network_cost'] <= params['R_W']].copy()
        
        # Adds prefix to water ID and converts type to string
        w_df['destination_id'] = 'W-' + w_df['destination_id'].apply(clean_id).astype(str)
        
        # Computes the penalty > uses clip instead of max. Clip works on the whole column at once > no need for loops
        w_df['penalty'] = ((w_df['network_cost'] - params['R_W_ideal']).clip(lower=0))**2
        
        # Creats a dict for the penalties and distances > sets index to origin and dest.
        penalty_water = clean_dict_keys(
            w_df.set_index(['origin_id', 'destination_id'])['penalty'].to_dict()
        )
        dist_water = clean_dict_keys(
            w_df.set_index(['origin_id', 'destination_id'])['network_cost'].to_dict()
        )
        
        # Creates list of all water IDs and updates the coordinate map
        W = list(w_df['destination_id'].unique())
        coord_map.update(extract_coords(w_df, 'destination_id'))

        # Load mobile OD matrix and does the same as the water OD matrix
        m_df = _read_csv_smart(mobile_od_file)
        
        m_df = m_df[m_df['network_cost'] <= params['R_M']].copy()
        
        m_df['destination_id'] = 'M-' + m_df['destination_id'].apply(clean_id).astype(str)
        
        bias = params.get('p_bias', 0)
        base_penalty = ((m_df['network_cost'] - params['R_M_ideal']).clip(lower=0))**2
        
        m_df['penalty_biased'] = base_penalty + bias
        m_df['penalty_raw'] = base_penalty 
        
        penalty_mobile = clean_dict_keys(
            m_df.set_index(['origin_id', 'destination_id'])['penalty_biased'].to_dict()
        )
        penalty_mobile_raw = clean_dict_keys(
            m_df.set_index(['origin_id', 'destination_id'])['penalty_raw'].to_dict()
        )
        dist_mobile = clean_dict_keys(
            m_df.set_index(['origin_id', 'destination_id'])['network_cost'].to_dict()
        )
        
        M = list(m_df['destination_id'].unique())
        coord_map.update(extract_coords(m_df, 'destination_id'))

        # Load underground-container OD matrices (one CSV per waste fraction).
        # u_od_files = {'rest': <file>, 'cardboard': <file>, 'glass': <file>}.
        # Any missing key is treated as "no UC option for this fraction" (empty sets).
        u_od_files_dict = u_od_files or {}
        U_t = {t: [] for t in WASTE_TYPES}
        penalty_underground_t = {t: {} for t in WASTE_TYPES}
        dist_underground_t = {t: {} for t in WASTE_TYPES}

        for t in WASTE_TYPES:
            u_file = u_od_files_dict.get(t)
            if u_file is None:
                continue

            u_df = _read_csv_smart(u_file)

            # R_U cutoff per fraction — supports sensitivity scans without rebuilding OD
            u_df = u_df[u_df['network_cost'] <= params['R_U'][t]].copy()

            # Prefix destination IDs so they're unique across fractions and modes
            prefix = f'U_{t}-'
            u_df['destination_id'] = prefix + u_df['destination_id'].apply(clean_id).astype(str)

            # Quadratic walking penalty above the ideal distance
            u_df['penalty'] = ((u_df['network_cost'] - params['R_U_ideal'][t]).clip(lower=0))**2

            penalty_underground_t[t] = clean_dict_keys(
                u_df.set_index(['origin_id', 'destination_id'])['penalty'].to_dict()
            )
            dist_underground_t[t] = clean_dict_keys(
                u_df.set_index(['origin_id', 'destination_id'])['network_cost'].to_dict()
            )
            U_t[t] = list(u_df['destination_id'].unique())
            # Coords for U destinations would come from a separate coord file if provided;
            # the OD CSV itself doesn't carry them.

        # Load extra coordinate files > not necessary to have them, but good to have
        if coord_files:
            if coord_files.get('demand'):
                coord_map.update(load_coords_from_file(coord_files['demand'], 'fid', prefix=''))
            if coord_files.get('water'):
                coord_map.update(load_coords_from_file(coord_files['water'], 'fid', prefix='W-'))
            if coord_files.get('mobile'):
                coord_map.update(load_coords_from_file(coord_files['mobile'], 'fid', prefix='M-'))

    except Exception as e:
        return None, f"Data Load Error: {e}"
        
    # Validate sets
    # sets are easier to work with because they are faster for lookups
    valid_I = set(I)
    
    # Only select pairs where all the data is present/availabe
    feasible_water_pairs = [
        (i, w) for (i, w) in penalty_water.keys() if i in valid_I
    ]
    feasible_mobile_pairs = [
        (i, m) for (i, m) in penalty_mobile.keys() if i in valid_I
    ]
    feasible_underground_pairs_t = {
        t: [(i, u) for (i, u) in penalty_underground_t[t].keys() if i in valid_I]
        for t in WASTE_TYPES
    }
    
    ####################################
    # NOT USED IN THESIS
    ####################################
    # Mobile restriction: Only allow mobile for households not covered by water
    # This is a seperate function that can be turned on or off
    if params.get('mobile_only_unreachable', False):
        # Get households that have at least one water facility within range
        households_with_water = set(i for (i, w) in feasible_water_pairs)
        
        # Filter mobile pairs to only include households NOT covered by water
        original_mobile_count = len(feasible_mobile_pairs)
        feasible_mobile_pairs = [
            (i, m) for (i, m) in feasible_mobile_pairs if i not in households_with_water
        ]
        
        # Also update the mobile facilities set M to only include facilities that still have feasible pairs
        used_mobile_facilities = set(m for (i, m) in feasible_mobile_pairs)
        original_M_count = len(M)
        M = [m for m in M if m in used_mobile_facilities]
        
        penalty_mobile = {k: v for k, v in penalty_mobile.items() if k in feasible_mobile_pairs}
        penalty_mobile_raw = {k: v for k, v in penalty_mobile_raw.items() if k in feasible_mobile_pairs}
        dist_mobile = {k: v for k, v in dist_mobile.items() if k in feasible_mobile_pairs}
    
    # Per-fraction coverage: each fraction t is independently assigned to W, M, or U_t.
    # A household is reachable iff EVERY fraction has at least one of:
    #   - a water facility in range,
    #   - a mobile facility in range, or
    #   - a UC for that specific fraction in range.
    has_w_set = {p[0] for p in feasible_water_pairs}
    has_m_set = {p[0] for p in feasible_mobile_pairs}
    has_u_set_per_t = {
        t: {p[0] for p in feasible_underground_pairs_t[t]} for t in WASTE_TYPES
    }
    covered_I = set(I)
    for t in WASTE_TYPES:
        covered_t = has_w_set | has_m_set | has_u_set_per_t[t]
        covered_I &= covered_t
    I_final = [i for i in I if i in covered_I]
    n_dropped = len(I) - len(I_final)
    if n_dropped:
        # Per-fraction breakdown of why each dropped household failed
        uncovered_by_t = {
            t: len(set(I) - (has_w_set | has_m_set | has_u_set_per_t[t]))
            for t in WASTE_TYPES
        }
        print(f"[DATA] Dropped {n_dropped} households missing coverage for at least "
              f"one fraction. Uncovered counts per fraction: {uncovered_by_t}")
    
    ####################################
    ####################################

    # get the first 5 values for possible error messages
    sample_demand_ids = list(I)[:5]
    sample_water_origins = list(set(k[0] for k in penalty_water.keys()))[:5]
    sample_mobile_origins = list(set(k[0] for k in penalty_mobile.keys()))[:5]
    
    if len(I_final) == 0:
        # DEBUG TIME > gives possible causes for the error
        msg = (
            f"❌ No households are within walking distance of any facility!\n\n"
            f"**Loaded counts:**\n"
            f"- Demand file: {len(I)} households\n"
            f"- Water OD: {len(W)} facilities, {len(penalty_water)} pairs\n"
            f"- Mobile OD: {len(M)} facilities, {len(penalty_mobile)} pairs\n"
            + "".join(
                f"- U_{t} OD: {len(U_t[t])} facilities, {len(penalty_underground_t[t])} pairs\n"
                for t in WASTE_TYPES
            )
            + f"\n**Sample IDs (check for mismatches):**\n"
            f"- Demand `fid`: `{sample_demand_ids}`\n"
            f"- Water `origin_id`: `{sample_water_origins}`\n"
            f"- Mobile `origin_id`: `{sample_mobile_origins}`\n\n"
            f"**Possible causes:**\n"
            f"1. ID type mismatch (string '123' vs integer 123)\n"
            f"2. Different ID columns (fid vs origin_id naming)\n"
            f"3. R_W={params['R_W']} / R_M={params['R_M']} / R_U={params['R_U']} too small\n"  # noqa: E501
            f"4. Wrong files uploaded"
        )
        return None, msg
    
    # Only keeps households/businesses that have at least 1 facility in range
    final_demand_data = {k: v for k, v in demand_data.items() if k[0] in I_final}

    # Trim per-fraction underground pairs to I_final too (mirrors what's implicit for W/M
    # via the valid_I filter above — keeping it explicit for the dict-of-sets case)
    final_I_set = set(I_final)
    feasible_underground_pairs_t = {
        t: [(i, u) for (i, u) in feasible_underground_pairs_t[t] if i in final_I_set]
        for t in WASTE_TYPES
    }

    return (
        I_final, W, M, U_t,
        feasible_water_pairs, feasible_mobile_pairs, feasible_underground_pairs_t,
        final_demand_data,
        penalty_water, penalty_mobile, penalty_mobile_raw, penalty_underground_t,
        coord_map, dist_water, dist_mobile, dist_underground_t,
    ), None
