import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import streamlit as st
import pulp

# Title
st.title("Delivery Cost and Route Optimization")

# Manual Entry of Deliveries
st.write("### Manual Entry of Deliveries")
D_a = st.number_input("Number of Type A deliveries (0-2 kg)", min_value=0, value=80)
D_b = st.number_input("Number of Type B deliveries (2-10 kg)", min_value=0, value=100)
D_c = st.number_input("Number of Type C deliveries (10-200 kg)", min_value=0, value=10)

# File uploader for Excel file
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Instructions for Excel Upload
st.write("""
### Instructions:
1. The Excel sheet must have column names in the first row.
2. The sheet to be analyzed must contain columns for 'Name', 'Latitude', 'Longitude', and 'Weight (KG)'.
3. The weights will be used to categorize deliveries into Type A (0-2 kg), Type B (2-10 kg), and Type C (10-200 kg).
""")

# Function to extract delivery data from the selected sheet
def extract_deliveries_from_excel(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    if 'Weight (KG)' not in df.columns:
        st.error("The selected sheet does not contain the required 'Weight (KG)' column.")
        return None
    
    return df

# Extract deliveries from uploaded Excel file
if uploaded_file:
    excel = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select Sheet", excel.sheet_names)
    if st.button("Extract Deliveries from Excel"):
        df_shops = extract_deliveries_from_excel(uploaded_file, sheet_name)
        if df_shops is not None:
            st.success("Deliveries extracted successfully.")

# Define the distance matrix calculation
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

# Define the nearest neighbor algorithm
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

# Define the global optimization function
def optimize_load_global(df, vehicle_info, scenario):
    vehicle_types = vehicle_info.keys()
    capacities = {v: vehicle_info[v]['capacity'] for v in vehicle_types}
    costs = {v: vehicle_info[v]['cost'] for v in vehicle_types}
    lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)
    vehicle_vars = {v: pulp.LpVariable(f'V_{v}', lowBound=0, cat='Integer') for v in vehicle_types}

    total_deliveries_a = sum((df['Weight (KG)'] > 0) & (df['Weight (KG)'] <= 2))
    total_deliveries_b = sum((df['Weight (KG)'] > 2) & (df['Weight (KG)'] <= 10))
    total_deliveries_c = sum((df['Weight (KG)'] > 10) & (df['Weight (KG)'] <= 200))
    total_weight_a = sum(df['Weight (KG)'][(df['Weight (KG)'] > 0) & (df['Weight (KG)'] <= 2)])
    total_weight_b = sum(df['Weight (KG)'][(df['Weight (KG)'] > 2) & (df['Weight (KG)'] <= 10)])
    total_weight_c = sum(df['Weight (KG)'][(df['Weight (KG)'] > 10) & (df['Weight (KG)'] <= 200)])

    lp_problem += pulp.lpSum([costs[v] * vehicle_vars[v] for v in vehicle_types]), "Total Cost"
    
    if scenario == "Scenario 1: V1, V2, V3":
        lp_problem += vehicle_vars['V1'] * 64 + vehicle_vars['V2'] * 66 + vehicle_vars['V3'] * 72 >= total_deliveries_a, "Delivery_Type_A_Constraint"
        lp_problem += vehicle_vars['V1'] * 64 + vehicle_vars['V2'] * 66 >= total_deliveries_b, "Delivery_Type_B_Constraint"
        lp_problem += vehicle_vars['V1'] * 64 >= total_deliveries_c, "Delivery_Type_C_Constraint"
        lp_problem += vehicle_vars['V1'] * capacities['V1'] >= total_weight_a + total_weight_b + total_weight_c, "Weight_Constraint_V1"
        lp_problem += vehicle_vars['V2'] * capacities['V2'] >= total_weight_b, "Weight_Constraint_V2"
        lp_problem += vehicle_vars['V3'] * capacities['V3'] >= total_weight_a, "Weight_Constraint_V3"

    elif scenario == "Scenario 2: V1, V2":
        lp_problem += vehicle_vars['V1'] * 64 + vehicle_vars['V2'] * 66 >= total_deliveries_a + total_deliveries_b, "Delivery_Type_A_B_Constraint"
        lp_problem += vehicle_vars['V1'] * 64 >= total_deliveries_c, "Delivery_Type_C_Constraint"
        lp_problem += vehicle_vars['V1'] * capacities['V1'] >= total_weight_a + total_weight_b + total_weight_c, "Weight_Constraint_V1"
        lp_problem += vehicle_vars['V2'] * capacities['V2'] >= total_weight_a + total_weight_b, "Weight_Constraint_V2"

    elif scenario == "Scenario 3: V1, V3":
        lp_problem += vehicle_vars['V1'] * 64 >= total_deliveries_b + total_deliveries_c, "Delivery_Type_B_C_Constraint"
        lp_problem += vehicle_vars['V3'] * 72 >= total_deliveries_a, "Delivery_Type_A_Constraint"
        lp_problem += vehicle_vars['V1'] * capacities['V1'] >= total_weight_b + total_weight_c, "Weight_Constraint_V1"
        lp_problem += vehicle_vars['V3'] * capacities['V3'] >= total_weight_a, "Weight_Constraint_V3"

    lp_problem.solve()
    status = pulp.LpStatus[lp_problem.status]
    vehicle_values = {v: pulp.value(vehicle_vars[v]) for v in vehicle_types}
    total_cost = pulp.value(lp_problem.objective)
    return status, vehicle_values, total_cost

# Display vehicle descriptions
vehicle_descriptions = {
    "V1": "A versatile vehicle capable of handling all types of deliveries with a higher cost and larger capacity (a four wheeler mini-truck).",
    "V2": "A mid-range vehicle that can handle types A and B deliveries with moderate cost and capacity (a three wheeler EV).",
    "V3": "A cost-effective vehicle that handles only type A deliveries with the smallest capacity (a two wheeler EV)."
}

st.subheader("Vehicle Information")
st.text("V1: " + vehicle_descriptions["V1"])
st.text("V2: " + vehicle_descriptions["V2"])
st.text("V3: " + vehicle_descriptions["V3"])

# User input for vehicle capacities
st.subheader("Vehicle Capacities (deliveries per day)")
v1_capacity = st.number_input("Capacity of V1", min_value=1, value=64)
v2_capacity = st.number_input("Capacity of V2", min_value=1, value=66)
v3_capacity = st.number_input("Capacity of V3", min_value=1, value=72)

# User input for vehicle costs
st.subheader("Vehicle Costs (INR per day)")
cost_v1 = st.number_input("Cost of V1", min_value=0.0, value=2416.0)
cost_v2 = st.number_input("Cost of V2", min_value=0.0, value=1270.0)
cost_v3 = st.number_input("Cost of V3", min_value=0.0, value=1115.0)

# User selection for scenario
scenario = st.selectbox("Select Scenario", ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"])

# Extract deliveries and optimize
if st.button("Optimize"):
    if uploaded_file:
        df_shops = extract_deliveries_from_excel(uploaded_file, sheet_name)
        if df_shops is not None:
            # Perform clustering and route optimization
            st.write("### Clustering and Route Optimization Results")
            distance_matrix = calculate_distance_matrix(df_shops)
            epsilon = 2000  # meters
            db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
            db.fit(distance_matrix)
            df_shops['Cluster'] = db.labels_
            st.write(df_shops)

            clusters = df_shops['Cluster'].unique()
            total_cost = 0
            vehicle_summary = {"V1": 0, "V2": 0, "V3": 0}

            for cluster_id in clusters:
                cluster = df_shops[df_shops['Cluster'] == cluster_id]
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

                route, cluster_total_distance = nearest_neighbor(cluster_distance_matrix)
                mapped_route = cluster.index[route]

                st.write(f"Cluster {cluster_id} Route:", " -> ".join(df_shops.loc[mapped_route, 'Name']))
                st.write(f"Cluster {cluster_id} Total Distance: {cluster_total_distance:.2f} meters")

                D_a = sum((cluster['Weight (KG)'] > 0) & (cluster['Weight (KG)'] <= 2))
                D_b = sum((cluster['Weight (KG)'] > 2) & (cluster['Weight (KG)'] <= 10))
                D_c = sum((cluster['Weight (KG)'] > 10) & (cluster['Weight (KG)'] <= 200))
                W_a = sum(cluster['Weight (KG)'][cluster['Weight (KG)'] <= 2])
                W_b = sum(cluster['Weight (KG)'][(cluster['Weight (KG)'] > 2) & (cluster['Weight (KG)'] <= 10)])
                W_c = sum(cluster['Weight (KG)'][cluster['Weight (KG)'] > 10])

                vehicle_info = {
                    'V1': {'capacity': v1_capacity, 'cost': cost_v1},
                    'V2': {'capacity': v2_capacity, 'cost': cost_v2},
                    'V3': {'capacity': v3_capacity, 'cost': cost_v3}
                }

                status, vehicle_values, cost = optimize_load_global(df_shops, vehicle_info, scenario)
                st.write(f"Load Optimization for Cluster {cluster_id}:")
                st.write(f"Status: {status}")
                st.write(f"Vehicles: {vehicle_values}")
                st.write(f"Total Cost: {cost:.2f}")

                total_cost += cost
                for v in vehicle_values:
                    vehicle_summary[v] += vehicle_values[v]

            st.write("### Summary")
            st.write(f"Total Cost: {total_cost:.2f}")
            st.write(f"Vehicle Summary: {vehicle_summary}")
