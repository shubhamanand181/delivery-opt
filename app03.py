import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp

# Title
st.title("Delivery Route and Load Optimization")

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
2. The sheet to be analyzed must contain columns named "Party", "Latitude", "Longitude", and "Weight (KG)".
3. The weights will be used to categorize deliveries into Type A (0-2 kg), Type B (2-10 kg), and Type C (10-200 kg).
""")

# Extract delivery data from the uploaded Excel file
df_locations = None
if uploaded_file:
    df_locations = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df_locations.head())

# Remove rows with NaN values in Latitude or Longitude
if df_locations is not None:
    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)

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

def calculate_distance_matrix(df):
    num_locations = len(df)
    distance_matrix = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                try:
                    coords_1 = (float(df.loc[i, 'Latitude']), float(df.loc[j, 'Longitude']))
                    coords_2 = (float(df.loc[j, 'Latitude']), float(df.loc[j, 'Longitude']))
                    distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                except ValueError as e:
                    print(f"Invalid coordinates at index {i} or {j}: {e}")
                    distance_matrix[i][j] = np.inf  # Assign a large value to indicate invalid distance
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

# Functions to run optimizations
def optimize_scenario_1(D_a, D_b, D_c, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity):
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
    lp_problem += A1 + A2 + A3 == D_a, "Total_Deliveries_A_Constraint"
    lp_problem += B1 + B2 == D_b, "Total_Deliveries_B_Constraint"
    lp_problem += C1 == D_c, "Total_Deliveries_C_Constraint"
    lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
    lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"
    lp_problem += v3_capacity * V3 >= A3, "V3_Capacity_Constraint"
    lp_problem += C1 == D_c, "Assign_C_To_V1"
    lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
    lp_problem += B2 == D_b - B1, "Assign_Remaining_B_To_V2"
    lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
    lp_problem += A2 <= v2_capacity * V2 - B2, "Assign_A_To_V2"
    lp_problem += A3 == D_a - A1 - A2, "Assign_Remaining_A_To_V3"
    lp_problem.solve()

    return {
        "Status": pulp.LpStatus[lp_problem.status],
        "V1": pulp.value(V1),
        "V2": pulp.value(V2),
        "V3": pulp.value(V3),
        "Total Cost": pulp.value(lp_problem.objective),
        "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
        "Deliveries assigned to V2": pulp.value(B2 + A2),
        "Deliveries assigned to V3": pulp.value(A3)
    }

def optimize_scenario_2(D_a, D_b, D_c, cost_v1, cost_v2, v1_capacity, v2_capacity):
    lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)

    V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
    V2 = pulp.LpVariable('V2', lowBound=0, cat='Integer')

    A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
    B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
    C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
    A2 = pulp.LpVariable('A2', lowBound=0, cat='Continuous')
    B2 = pulp.LpVariable('B2', lowBound=0, cat='Continuous')

    lp_problem += cost_v1 * V1 + cost_v2 * V2, "Total Cost"

    lp_problem += A1 + A2 == D_a, "Total_Deliveries_A_Constraint"
    lp_problem += B1 + B2 == D_b, "Total_Deliveries_B_Constraint"
    lp_problem += C1 == D_c, "Total_Deliveries_C_Constraint"

    lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
    lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"

    lp_problem += C1 == D_c, "Assign_C_To_V1"
    lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
    lp_problem += B2 == D_b - B1, "Assign_Remaining_B_To_V2"
    lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
    lp_problem += A2 == D_a - A1, "Assign_Remaining_A_To_V2"

    lp_problem.solve()

    return {
        "Status": pulp.LpStatus[lp_problem.status],
        "V1": pulp.value(V1),
        "V2": pulp.value(V2),
        "Total Cost": pulp.value(lp_problem.objective),
        "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
        "Deliveries assigned to V2": pulp.value(B2 + A2)
    }

def optimize_scenario_3(D_a, D_b, D_c, cost_v1, cost_v3, v1_capacity, v3_capacity):
    lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)

    V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
    V3 = pulp.LpVariable('V3', lowBound=0, cat='Integer')

    A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
    B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
    C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
    A3 = pulp.LpVariable('A3', lowBound=0, cat='Continuous')

    lp_problem += cost_v1 * V1 + cost_v3 * V3, "Total Cost"

    lp_problem += A1 + A3 == D_a, "Total_Deliveries_A_Constraint"
    lp_problem += B1 == D_b, "Total_Deliveries_B_Constraint"
    lp_problem += C1 == D_c, "Total_Deliveries_C_Constraint"

    lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
    lp_problem += v3_capacity * V3 >= A3, "V3_Capacity_Constraint"

    lp_problem += C1 == D_c, "Assign_C_To_V1"
    lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
    lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
    lp_problem += A3 == D_a - A1, "Assign_Remaining_A_To_V3"

    lp_problem.solve()

    return {
        "Status": pulp.LpStatus[lp_problem.status],
        "V1": pulp.value(V1),
        "V3": pulp.value(V3),
        "Total Cost": pulp.value(lp_problem.objective),
        "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
        "Deliveries assigned to V3": pulp.value(A3)
    }

# Perform optimization when the button is clicked
if st.button("Optimize"):
    if df_locations is not None:
        # Calculate the distance matrix
        distance_matrix = calculate_distance_matrix(df_locations)

        # Apply DBSCAN
        epsilon = 100  # meters
        db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
        db.fit(distance_matrix)

        # Add cluster labels to the DataFrame
        df_locations['Cluster'] = db.labels_
        st.write("Clustered Data:")
        st.write(df_locations)

        # Calculate centroids of clusters
        centroids = df_locations.groupby('Cluster')[['Latitude', 'Longitude']].mean()
        st.write("Centroids of Clusters:")
        st.write(centroids)

        # Calculate distance matrix for each cluster and plan routes
        clusters = df_locations['Cluster'].unique()
        for cluster_id in clusters:
            cluster = df_locations[df_locations['Cluster'] == cluster_id]
            num_locations = len(cluster)

            # Create distance matrix
            distance_matrix = np.zeros((num_locations, num_locations))
            for i in range(num_locations):
                for j in range(num_locations):
                    if i != j:
                        coords_1 = (cluster.iloc[i]['Latitude'], cluster.iloc[j]['Longitude'])
                        coords_2 = (cluster.iloc[j]['Latitude'], cluster.iloc[j]['Longitude'])
                        distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                    else:
                        distance_matrix[i][j] = np.inf  # Use infinity to avoid self-loops

            # Optimize route for the cluster
            route, total_distance = nearest_neighbor(distance_matrix)
            mapped_route = cluster.index[route]

            st.write(f"Cluster {cluster_id} Route:", " -> ".join(df_locations.loc[mapped_route, 'Party']))
            st.write(f"Cluster {cluster_id} Total Distance: {total_distance / 1000:.2f} kilometers")

        # Create distance matrix for centroids
        num_clusters = len(centroids)
        centroid_distance_matrix = np.zeros((num_clusters, num_clusters))

        for i in range(num_clusters):
            for j in range(num_clusters):
                if i != j:
                    coords_1 = (centroids.iloc[i]['Latitude'], centroids.iloc[j]['Longitude'])
                    coords_2 = (centroids.iloc[j]['Latitude'], centroids.iloc[j]['Longitude'])
                    centroid_distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                else:
                    centroid_distance_matrix[i][j] = np.inf

        # Optimize route for the centroids (clusters)
        centroid_route, centroid_total_distance = nearest_neighbor(centroid_distance_matrix)
        st.write("Cluster Sequence Route:", " -> ".join(map(str, centroid_route)))
        st.write(f"Total Distance for Clusters: {centroid_total_distance / 1000:.2f} kilometers")

        # Load Optimization
        D_a = sum((df_locations['Weight (KG)'] > 0) & (df_locations['Weight (KG)'] <= 2))
        D_b = sum((df_locations['Weight (KG)'] > 2) & (df_locations['Weight (KG)'] <= 10))
        D_c = sum((df_locations['Weight (KG)'] > 10) & (df_locations['Weight (KG)'] <= 200))

        if scenario == "Scenario 1: V1, V2, V3":
            result = optimize_scenario_1(D_a, D_b, D_c, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity)
        elif scenario == "Scenario 2: V1, V2":
            result = optimize_scenario_2(D_a, D_b, D_c, cost_v1, cost_v2, v1_capacity, v2_capacity)
        elif scenario == "Scenario 3: V1, V3":
            result = optimize_scenario_3(D_a, D_b, D_c, cost_v1, cost_v3, v1_capacity, v3_capacity)

        st.write("Load Optimization Results:")
        st.write(f"Status: {result['Status']}")
        st.write(f"V1: {result['V1']}")
        if "V2" in result:
            st.write(f"V2: {result['V2']}")
        if "V3" in result:
            st.write(f"V3: {result['V3']}")
        st.write(f"Total Cost: {result['Total Cost']}")
        st.write(f"Deliveries assigned to V1: {result['Deliveries assigned to V1']}")
        if "Deliveries assigned to V2" in result:
            st.write(f"Deliveries assigned to V2: {result['Deliveries assigned to V2']}")
        if "Deliveries assigned to V3" in result:
            st.write(f"Deliveries assigned to V3: {result['Deliveries assigned to V3']}")
    else:
        st.write("Please upload an Excel file with the required columns.")
