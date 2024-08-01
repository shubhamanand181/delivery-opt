import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp

def extract_deliveries_from_excel(file):
    df = pd.read_excel(file)
    expected_columns = ['Party', 'Latitude', 'Longitude', 'Weight (KG)']
    if all(col in df.columns for col in expected_columns):
        df = df.dropna(subset=['Latitude', 'Longitude'])
        return df
    else:
        st.error("One or more expected columns are missing. Please check the column names in the Excel file.")
        return None

def categorize_weights(df):
    D_a = df[(df['Weight (KG)'] > 0) & (df['Weight (KG)'] <= 2)]
    D_b = df[(df['Weight (KG)'] > 2) & (df['Weight (KG)'] <= 10)]
    D_c = df[(df['Weight (KG)'] > 10) & (df['Weight (KG)'] <= 200)]
    return D_a, D_b, D_c

def calculate_distance_matrix(df):
    num_locations = len(df)
    distance_matrix = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                coords_1 = (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])
                coords_2 = (df.loc[j, 'Latitude'], df.loc[j, 'Longitude'])
                distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
            else:
                distance_matrix[i][j] = 0
    return distance_matrix

def nearest_neighbor(distance_matrix, start_index=0):
    num_locations = len(distance_matrix)
    visited = [False] * num_locations
    route = [start_index]
    total_distance = 0

    current_index = start_index
    visited[current_index] = True

    for _ in range(num_locations - 1):
        next_index = None
        min_distance = np.inf

        for j in range(num_locations):
            if not visited[j] and distance_matrix[current_index][j] < min_distance:
                next_index = j
                min_distance = distance_matrix[current_index][j]

        route.append(next_index)
        total_distance += min_distance
        current_index = next_index
        visited[current_index] = True

    # Return to the start point
    route.append(start_index)
    total_distance += distance_matrix[current_index][start_index]

    return route, total_distance

def optimize_load(D_a_count, D_b_count, D_c_count, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario):
    if scenario == "Scenario 1: V1, V2, V3":
        lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)
        V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
        V2 = pulp.LpVariable('V2', lowBound=0, cat='Integer')
        V3 = pulp.LpVariable('V3', lowBound=0, cat='Integer')

        A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
        B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
        C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
        A2 = pulp.LpVariable('A2', lowBound=0, cat='Continuous')
        B2 = pulp.LpVariable('B2', lowBound=0, cat='Continuous')
        A3 = pulp.LpVariable('A3', lowBound=0, cat='Continuous')

        lp_problem += cost_v1 * V1 + cost_v2 * V2 + cost_v3 * V3, "Total Cost"
        lp_problem += A1 + A2 + A3 == D_a_count, "Total_Deliveries_A_Constraint"
        lp_problem += B1 + B2 == D_b_count, "Total_Deliveries_B_Constraint"
        lp_problem += C1 == D_c_count, "Total_Deliveries_C_Constraint"
        lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
        lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"
        lp_problem += v3_capacity * V3 >= A3, "V3_Capacity_Constraint"
        lp_problem += C1 == D_c_count, "Assign_C_To_V1"
        lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
        lp_problem += B2 == D_b_count - B1, "Assign_Remaining_B_To_V2"
        lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
        lp_problem += A2 <= v2_capacity * V2 - B2, "Assign_A_To_V2"
        lp_problem += A3 == D_a_count - A1 - A2, "Assign_Remaining_A_To_V3"
        lp_problem.solve()

    elif scenario == "Scenario 2: V1, V2":
        lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)
        V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
        V2 = pulp.LpVariable('V2', lowBound=0, cat='Integer')

        A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
        B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
        C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
        A2 = pulp.LpVariable('A2', lowBound=0, cat='Continuous')
        B2 = pulp.LpVariable('B2', lowBound=0, cat='Continuous')

        lp_problem += cost_v1 * V1 + cost_v2 * V2, "Total Cost"
        lp_problem += A1 + A2 == D_a_count, "Total_Deliveries_A_Constraint"
        lp_problem += B1 + B2 == D_b_count, "Total_Deliveries_B_Constraint"
        lp_problem += C1 == D_c_count, "Total_Deliveries_C_Constraint"
        lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
        lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"
        lp_problem += C1 == D_c_count, "Assign_C_To_V1"
        lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
        lp_problem += B2 == D_b_count - B1, "Assign_Remaining_B_To_V2"
        lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
        lp_problem += A2 == D_a_count - A1, "Assign_Remaining_A_To_V2"
        lp_problem.solve()

    elif scenario == "Scenario 3: V1, V3":
        lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)
        V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
        V3 = pulp.LpVariable('V3', lowBound=0, cat='Integer')

        A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
        B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
        C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
        A3 = pulp.LpVariable('A3', lowBound=0, cat='Continuous')

        lp_problem += cost_v1 * V1 + cost_v3 * V3, "Total Cost"
        lp_problem += A1 + A3 == D_a_count, "Total_Deliveries_A_Constraint"
        lp_problem += B1 == D_b_count, "Total_Deliveries_B_Constraint"
        lp_problem += C1 == D_c_count, "Total_Deliveries_C_Constraint"
        lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
        lp_problem += v3_capacity * V3 >= A3, "V3_Capacity_Constraint"
        lp_problem += C1 == D_c_count, "Assign_C_To_V1"
        lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
        lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
        lp_problem += A3 == D_a_count - A1, "Assign_Remaining_A_To_V3"
        lp_problem.solve()

    return {
        "Status": pulp.LpStatus[lp_problem.status],
        "V1": pulp.value(V1),
        "V2": pulp.value(V2) if scenario in ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2"] else None,
        "V3": pulp.value(V3) if scenario in ["Scenario 1: V1, V2, V3", "Scenario 3: V1, V3"] else None,
        "Total Cost": pulp.value(lp_problem.objective),
        "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
        "Deliveries assigned to V2": pulp.value(B2 + A2) if scenario in ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2"] else None,
        "Deliveries assigned to V3": pulp.value(A3) if scenario in ["Scenario 1: V1, V2, V3", "Scenario 3: V1, V3"] else None
    }

def generate_routes(vehicle_assignments, df_locations):
    vehicle_routes = {"V1": [], "V2": [], "V3": []}
    for vehicle, indexes in vehicle_assignments.items():
        df_vehicle = df_locations.loc[indexes].reset_index(drop=True)
        st.write(f"Processing {vehicle} with {len(df_vehicle)} locations")

        if df_vehicle.empty:
            st.write(f"No locations for {vehicle}")
            continue

        # Calculate the distance matrix for the vehicle
        distance_matrix = calculate_distance_matrix(df_vehicle)
        epsilon = 500  # meters for DBSCAN clustering
        db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
        db.fit(distance_matrix)
        df_vehicle['Cluster'] = db.labels_

        # Calculate centroids of clusters
        centroids = df_vehicle.groupby('Cluster')[['Latitude', 'Longitude']].mean().reset_index()
        vehicle_routes[vehicle].append({"Route": df_vehicle, "Centroids": centroids})

    return vehicle_routes

def display_summary(vehicle_routes):
    summary_data = []
    for vehicle, clusters in vehicle_routes.items():
        for cluster in clusters:
            centroid = cluster['Centroids'].iloc[0]
            total_distance = cluster['Route']['Distance'].sum() if 'Distance' in cluster['Route'].columns else 0
            summary_data.append({
                "Cluster": cluster['Centroids']['Cluster'].iloc[0],
                "Vehicle": vehicle,
                "Latitude": centroid['Latitude'],
                "Longitude": centroid['Longitude'],
                "Number of Shops": len(cluster['Route']),
                "Total Distance": total_distance
            })
    summary_df = pd.DataFrame(summary_data)
    st.write(summary_df)
    return summary_df

def generate_excel(vehicle_routes, summary_df):
    with pd.ExcelWriter('/mnt/data/optimized_routes.xlsx', engine='xlsxwriter') as writer:
        for vehicle, clusters in vehicle_routes.items():
            for cluster in clusters:
                df_cluster = pd.DataFrame(cluster['Route'])
                df_cluster.to_excel(writer, sheet_name=f"{vehicle}_Cluster_{cluster['Cluster']}", index=False)
        
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

# Streamlit UI
st.title("Delivery Optimization App")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    df_locations = extract_deliveries_from_excel(uploaded_file)
    
    if df_locations is not None:
        # User input for vehicle costs and capacities
        st.subheader("Vehicle Cost and Capacity")
        cost_v1 = st.number_input("Cost of Vehicle Type V1", value=62.8156)
        cost_v2 = st.number_input("Cost of Vehicle Type V2", value=33.0)
        cost_v3 = st.number_input("Cost of Vehicle Type V3", value=29.0536)
        v1_capacity = st.number_input("Capacity of Vehicle Type V1", value=64)
        v2_capacity = st.number_input("Capacity of Vehicle Type V2", value=66)
        v3_capacity = st.number_input("Capacity of Vehicle Type V3", value=72)
        scenario = st.selectbox("Choose Scenario", ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"])
        
        if st.button("Optimize Load"):
            D_a, D_b, D_c = categorize_weights(df_locations)
            result = optimize_load(len(D_a), len(D_b), len(D_c), cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario)
            
            st.write("Load Optimization Results:")
            st.write(f"Status: {result['Status']}")
            st.write(f"V1: {result['V1']}")
            st.write(f"V2: {result['V2']}")
            st.write(f"V3: {result['V3']}")
            st.write(f"Total Cost: {result['Total Cost']}")
            st.write(f"Deliveries assigned to V1: {result['Deliveries assigned to V1']}")
            st.write(f"Deliveries assigned to V2: {result['Deliveries assigned to V2']}")
            st.write(f"Deliveries assigned to V3: {result['Deliveries assigned to V3']}")

            # Assign deliveries to vehicles
            vehicle_assignments = {
                "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
                "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):]))].tolist(),
                "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):])):].tolist()
            }

            st.session_state.vehicle_assignments = vehicle_assignments

        if st.button("Generate Routes") and 'vehicle_assignments' in st.session_state:
            vehicle_assignments = st.session_state.vehicle_assignments
            vehicle_routes = generate_routes(vehicle_assignments, df_locations)
            summary_df = display_summary(vehicle_routes)
            generate_excel(vehicle_routes, summary_df)

            st.write("Routes and Summary saved to Excel file.")
            st.write("[Download Excel File](optimized_routes.xlsx)")
