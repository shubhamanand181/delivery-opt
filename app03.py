import streamlit as st
import pandas as pd
import pulp
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle

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
2. The sheet to be analyzed must contain columns named 'Party', 'Latitude', 'Longitude', and 'Weight (KG)'.
3. The weights will be used to categorize deliveries into Type A (0-2 kg), Type B (2-10 kg), and Type C (10-200 kg).
""")

# Function to extract delivery data from the selected sheet
def extract_deliveries_from_excel(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    required_columns = ['Party', 'Latitude', 'Longitude', 'Weight (KG)']
    if not all(column in df.columns for column in required_columns):
        st.error(f"The selected sheet does not contain the required columns: {', '.join(required_columns)}")
        return None, None, None, None
    
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    weight = df['Weight (KG)']
    D_a = sum((weight > 0) & (weight <= 2))
    D_b = sum((weight > 2) & (weight <= 10))
    D_c = sum((weight > 10) & (weight <= 200))
    
    return df, D_a, D_b, D_c

# Extract deliveries from uploaded Excel file
if uploaded_file:
    excel = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select Sheet", excel.sheet_names)
    if st.button("Extract Deliveries from Excel"):
        df_shops, D_a, D_b, D_c = extract_deliveries_from_excel(uploaded_file, sheet_name)
        if df_shops is not None:
            st.success(f"Extracted Deliveries - Type A: {D_a}, Type B: {D_b}, Type C: {D_c}")
            st.dataframe(df_shops)

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

# Route Optimization Functions
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

# Route Optimization
if df_shops is not None:
    distance_matrix = calculate_distance_matrix(df_shops)
    epsilon = 100  # meters
    db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
    db.fit(distance_matrix)
    df_shops['Cluster'] = db.labels_
    centroids = df_shops.groupby('Cluster')[['Latitude', 'Longitude']].mean()
    clusters = df_shops['Cluster'].unique()
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
        route, total_distance = nearest_neighbor(cluster_distance_matrix)
        mapped_route = cluster.index[route]
        st.write(f"Cluster {cluster_id} Route:", " -> ".join(df_shops.loc[mapped_route, 'Party']))
        st.write(f"Cluster {cluster_id} Total Distance: {total_distance / 1000:.2f} kilometers")

    # Optimize route for the centroids (clusters)
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
    centroid_route, centroid_total_distance = nearest_neighbor(centroid_distance_matrix)
    st.write("Cluster Sequence Route:", " -> ".join(map(str, centroid_route)))
    st.write(f"Total Distance for Clusters: {centroid_total_distance / 1000:.2f} kilometers")

# Load Optimization
def optimize_load(D_a, D_b, D_c, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario):
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

# Example Load Optimization with extracted data
if df_shops is not None:
    D_a = sum((df_shops['Weight (KG)'] > 0) & (df_shops['Weight (KG)'] <= 2))
    D_b = sum((df_shops['Weight (KG)'] > 2) & (df_shops['Weight (KG)'] <= 10))
    D_c = sum((df_shops['Weight (KG)'] > 10) & (df_shops['Weight (KG)'] <= 200))

    # Optimize load
    result = optimize_load(D_a, D_b, D_c, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario)
    st.write("Load Optimization Results:")
    st.write(f"Status: {result['Status']}")
    st.write(f"V1: {result['V1']}")
    st.write(f"V2: {result['V2']}")
    st.write(f"V3: {result['V3']}")
    st.write(f"Total Cost: {result['Total Cost']}")
    st.write(f"Deliveries assigned to V1: {result['Deliveries assigned to V1']}")
    st.write(f"Deliveries assigned to V2: {result['Deliveries assigned to V2']}")
    st.write(f"Deliveries assigned to V3: {result['Deliveries assigned to V3']}")
