import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp
import os

# Streamlit app title
st.title("Delivery Optimization App")

# File uploader for Excel file
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Instructions for Excel Upload
st.write("""
### Instructions:
1. The Excel sheet must have column names in the first row.
2. The sheet to be analyzed must contain columns named "Party", "Latitude", "Longitude", and "Weight (KG)".
""")

# Define the load optimization function
def optimize_load(D_a_count, D_b_count, D_c_count, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario):
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

    if scenario == "Scenario 1: V1, V2, V3":
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
    elif scenario == "Scenario 2: V1, V2":
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
    elif scenario == "Scenario 3: V1, V3":
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
        "V2": pulp.value(V2) if scenario != "Scenario 3: V1, V3" else None,
        "V3": pulp.value(V3) if scenario != "Scenario 2: V1, V2" else None,
        "Total Cost": pulp.value(lp_problem.objective),
        "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
        "Deliveries assigned to V2": pulp.value(B2 + A2) if scenario != "Scenario 3: V1, V3" else None,
        "Deliveries assigned to V3": pulp.value(A3) if scenario != "Scenario 2: V1, V2" else None
    }

# Function to categorize weights
def categorize_weights(df):
    D_a = df[(df['Weight (KG)'] > 0) & (df['Weight (KG)'] <= 2)]
    D_b = df[(df['Weight (KG)'] > 2) & (df['Weight (KG)'] <= 10)]
    D_c = df[(df['Weight (KG)'] > 10) & (df['Weight (KG)'] <= 200)]
    return D_a, D_b, D_c

# Function to calculate distance matrix
def calculate_distance_matrix(df):
    num_locations = len(df)
    distance_matrix = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                try:
                    coords_1 = (float(df.iloc[i]['Latitude']), float(df.iloc[i]['Longitude']))
                    coords_2 = (float(df.iloc[j]['Latitude']), float(df.iloc[j]['Longitude']))
                    distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                except ValueError as e:
                    st.write(f"Invalid coordinates at index {i} or {j}: {e}")
                    distance_matrix[i][j] = np.inf  # Assign a large value to indicate invalid distance
            else:
                distance_matrix[i][j] = 0
    return distance_matrix

# Function to perform nearest neighbor route optimization
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

# Function to generate routes based on vehicle assignments
def generate_routes(vehicle_assignments, df_locations):
    vehicle_routes = {}
    summary_data = []

    for vehicle, indices in vehicle_assignments.items():
        df_vehicle = df_locations.loc[indices]

        if df_vehicle.empty:
            st.write(f"No deliveries assigned to {vehicle}.")
            continue

        distance_matrix = calculate_distance_matrix(df_vehicle)
        
        # Ensure no invalid distances
        if np.isinf(distance_matrix).any() or np.isnan(distance_matrix).any():
            st.write(f"Invalid distance matrix for {vehicle}")
            continue

        db = DBSCAN(eps=100, min_samples=1, metric='precomputed')
        db.fit(distance_matrix)
        df_vehicle['Cluster'] = db.labels_

        clusters = df_vehicle['Cluster'].unique()
        vehicle_clusters = []

        for cluster_id in clusters:
            cluster = df_vehicle[df_vehicle['Cluster'] == cluster_id]
            num_locations = len(cluster)
            cluster_distance_matrix = np.zeros((num_locations, num_locations))

            for i in range(num_locations):
                for j in range(num_locations):
                    if i != j:
                        coords_1 = (cluster.iloc[i]['Latitude'], cluster.iloc[j]['Longitude'])
                        coords_2 = (cluster.iloc[j]['Latitude'], cluster.iloc[j]['Longitude'])
                        cluster_distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                    else:
                        cluster_distance_matrix[i][j] = np.inf

            # Optimize route for the cluster
            route, total_distance = nearest_neighbor(cluster_distance_matrix)
            mapped_route = cluster.iloc[route]

            summary_data.append({
                'Cluster': cluster_id,
                'Vehicle': vehicle,
                'Number of Shops': len(cluster),
                'Total Distance': total_distance / 1000,
                'Latitude': cluster['Latitude'].mean(),
                'Longitude': cluster['Longitude'].mean()
            })

            vehicle_clusters.append({
                'Route': mapped_route.to_dict(orient='records'),
                'Centroids': cluster[['Cluster', 'Latitude', 'Longitude']].mean()
            })

        vehicle_routes[vehicle] = vehicle_clusters

    summary_df = pd.DataFrame(summary_data)
    return vehicle_routes, summary_df

# Function to generate Excel file with routes and summary
def generate_excel(vehicle_routes, summary_df):
    file_path = 'optimized_routes.xlsx'
    
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for vehicle, clusters in vehicle_routes.items():
            for cluster in clusters:
                df_cluster = pd.DataFrame(cluster['Route'])
                df_cluster.to_excel(writer, sheet_name=f"{vehicle}_Cluster_{int(cluster['Centroids']['Cluster'])}", index=False)

        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    st.write("Routes and Summary saved to Excel file.")
    with open(file_path, 'rb') as file:
        st.download_button(label='Download Excel File', data=file, file_name='optimized_routes.xlsx')

if uploaded_file is not None:
    df_locations = pd.read_excel(uploaded_file)
    
    # Display the column names to verify
    st.write("Column Names:", df_locations.columns)

    # Check the first few rows of the DataFrame
    st.write(df_locations.head())

    # Ensure column names are as expected
    expected_columns = ['Party', 'Latitude', 'Longitude', 'Weight (KG)']
    if all(col in df_locations.columns for col in expected_columns):
        st.write("All expected columns are present.")
    else:
        st.write("One or more expected columns are missing. Please check the column names in the Excel file.")

    # Remove rows with NaN values in Latitude or Longitude
    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # Inputs for Load Optimization
    st.subheader("Load Optimization")
    scenario = st.selectbox("Select Scenario", ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"])
    cost_v1 = st.number_input("Cost for Vehicle V1", value=62.8156)
    cost_v2 = st.number_input("Cost for Vehicle V2", value=33.0)
    cost_v3 = st.number_input("Cost for Vehicle V3", value=29.0536)
    v1_capacity = st.number_input("Capacity of Vehicle V1", value=64)
    v2_capacity = st.number_input("Capacity of Vehicle V2", value=66)
    v3_capacity = st.number_input("Capacity of Vehicle V3", value=72)

    if st.button("Optimize Load"):
        D_a, D_b, D_c = categorize_weights(df_locations)
        result = optimize_load(len(D_a), len(D_b), len(D_c), cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario)

        st.write("Load Optimization Results:")
        st.write(f"Status: {result['Status']}")
        st.write(f"V1: {result['V1']}")
        if result['V2'] is not None:
            st.write(f"V2: {result['V2']}")
        if result['V3'] is not None:
            st.write(f"V3: {result['V3']}")
        st.write(f"Total Cost: {result['Total Cost']}")
        st.write(f"Deliveries assigned to V1: {result['Deliveries assigned to V1']}")
        if result['Deliveries assigned to V2'] is not None:
            st.write(f"Deliveries assigned to V2: {result['Deliveries assigned to V2']}")
        if result['Deliveries assigned to V3'] is not None:
            st.write(f"Deliveries assigned to V3: {result['Deliveries assigned to V3']}")

        # Assign deliveries to vehicles
        vehicle_assignments = {
            "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
            "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + (result.get('Deliveries assigned to V2') or 0) - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):]))].tolist() if result['Deliveries assigned to V2'] is not None else [],
            "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + (result.get('Deliveries assigned to V2') or 0) - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):])):].tolist() if result['Deliveries assigned to V3'] is not None else []
        }

        st.session_state.vehicle_assignments = vehicle_assignments

    if st.button("Generate Routes") and 'vehicle_assignments' in st.session_state:
        vehicle_assignments = st.session_state.vehicle_assignments
        vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)
        generate_excel(vehicle_routes, summary_df)
