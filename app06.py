import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp

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
3. The weights will be used to categorize deliveries into Type A (0-2 kg), Type B (2-10 kg), and Type C (10-200 kg).
""")

# Function to extract delivery data from the selected sheet
def extract_deliveries_from_excel(file):
    df = pd.read_excel(file)
    if not all(col in df.columns for col in ['Party', 'Latitude', 'Longitude', 'Weight (KG)']):
        st.error("The selected sheet does not contain the required columns.")
        return None
    
    # Remove rows with NaN values in Latitude or Longitude
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    return df

# Categorize weights
def categorize_weights(df):
    D_a = df[(df['Weight (KG)'] > 0) & (df['Weight (KG)'] <= 2)]
    D_b = df[(df['Weight (KG)'] > 2) & (df['Weight (KG)'] <= 10)]
    D_c = df[(df['Weight (KG)'] > 10) & (df['Weight (KG)'] <= 200)]
    return D_a, D_b, D_c

# Load optimization function
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
        "V1": pulp.value(V1) if 'V1' in locals() else 0,
        "V2": pulp.value(V2) if 'V2' in locals() else 0,
        "V3": pulp.value(V3) if 'V3' in locals() else 0,
        "Total Cost": pulp.value(lp_problem.objective),
        "Deliveries assigned to V1": pulp.value(C1) + pulp.value(B1) + pulp.value(A1) if 'C1' in locals() else 0,
        "Deliveries assigned to V2": pulp.value(B2) + pulp.value(A2) if 'B2' in locals() else 0,
        "Deliveries assigned to V3": pulp.value(A3) if 'A3' in locals() else 0
    }

# Generate routes function
def generate_routes(vehicle_assignments, df_locations):
    vehicle_routes = {}
    epsilon = 500  # meters

    for vehicle, indices in vehicle_assignments.items():
        df_vehicle = df_locations.loc[indices].reset_index(drop=True)

        if df_vehicle.empty:
            continue

        distance_matrix = calculate_distance_matrix(df_vehicle)
        db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
        db.fit(distance_matrix)
        df_vehicle['Cluster'] = db.labels_

        centroids = df_vehicle.groupby('Cluster')[['Latitude', 'Longitude']].mean()
        cluster_routes = []

        for cluster_id, cluster_data in df_vehicle.groupby('Cluster'):
            cluster_indices = cluster_data.index.tolist()
            cluster_distance_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]

            route, total_distance = nearest_neighbor(cluster_distance_matrix)
            mapped_route = cluster_data.index[route]

            cluster_routes.append({
                "Cluster": cluster_id,
                "Route": df_vehicle.loc[mapped_route].to_dict('records'),
                "Total Distance": total_distance / 1000  # in kilometers
            })

        vehicle_routes[vehicle] = cluster_routes

    return vehicle_routes

# Summary display function
def display_summary(vehicle_routes):
    summary_data = []
    for vehicle, clusters in vehicle_routes.items():
        for cluster in clusters:
            total_distance = cluster['Total Distance']
            centroid = cluster['Route'][0]  # Assume the first point as centroid for simplicity
            summary_data.append({
                "Vehicle": vehicle,
                "Cluster": cluster['Cluster'],
                "Latitude": centroid['Latitude'],
                "Longitude": centroid['Longitude'],
                "Number of Shops": len(cluster['Route']),
                "Total Distance": total_distance
            })
    summary_df = pd.DataFrame(summary_data)
    st.write(summary_df)
    return summary_df

# Excel generation function
def generate_excel(vehicle_routes, summary_df):
    with pd.ExcelWriter('/mnt/data/optimized_routes.xlsx', engine='xlsxwriter') as writer:
        for vehicle, clusters in vehicle_routes.items():
            for cluster in clusters:
                df_cluster = pd.DataFrame(cluster['Route'])
                df_cluster.to_excel(writer, sheet_name=f"{vehicle}_Cluster_{cluster['Cluster']}", index=False)
        
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

# Load and route optimization section
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
