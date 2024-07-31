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
                    st.error(f"Invalid coordinates at index {i} or {j}: {e}")
                    distance_matrix[i][j] = np.inf  # Assign a large value to indicate invalid distance
            else:
                distance_matrix[i][j] = 0
    return distance_matrix

# Function to generate routes and clusters
def generate_routes(vehicle_assignments, df):
    vehicle_routes = {}

    for vehicle, assignments in vehicle_assignments.items():
        vehicle_df = df.loc[assignments]
        distance_matrix = calculate_distance_matrix(vehicle_df)

        # Define the maximum distance for points to be considered in the same cluster
        epsilon = 500  # meters

        # Apply DBSCAN
        db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
        db.fit(distance_matrix)

        # Add cluster labels to the DataFrame
        vehicle_df['Cluster'] = db.labels_
        
        vehicle_routes[vehicle] = vehicle_df
        
    return vehicle_routes

# Function to display summary
def display_summary(vehicle_routes):
    summary_data = []

    for vehicle, df in vehicle_routes.items():
        clusters = df['Cluster'].unique()
        for cluster_id in clusters:
            cluster_df = df[df['Cluster'] == cluster_id]
            centroid_lat = cluster_df['Latitude'].mean()
            centroid_long = cluster_df['Longitude'].mean()
            num_shops = len(cluster_df)
            total_distance = 0  # Calculate based on route optimization (omitted for brevity)
            
            summary_data.append({
                "Cluster": cluster_id,
                "Latitude": centroid_lat,
                "Longitude": centroid_long,
                "Vehicle": vehicle,
                "Number of Shops": num_shops,
                "Total Distance": total_distance
            })

    summary_df = pd.DataFrame(summary_data)
    st.write("Summary of Clusters")
    st.dataframe(summary_df)
    return summary_df

# Function to generate Excel file
def generate_excel(vehicle_routes, summary_df):
    with pd.ExcelWriter('/mnt/data/optimized_routes.xlsx', engine='xlsxwriter') as writer:
        for vehicle, df in vehicle_routes.items():
            df.to_excel(writer, sheet_name=f'{vehicle}_routes', index=False)
        
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    st.write("Download the optimized routes and summary:")
    st.download_button(label="Download Excel file", data=open('/mnt/data/optimized_routes.xlsx', 'rb'), file_name='optimized_routes.xlsx')

# Main code
if uploaded_file:
    df_locations = extract_deliveries_from_excel(uploaded_file)
    if df_locations is not None:
        D_a, D_b, D_c = categorize_weights(df_locations)
        
        st.write("### Load Optimization")
        cost_v1 = st.number_input("Cost of V1", min_value=0.0, value=62.8156)
        cost_v2 = st.number_input("Cost of V2", min_value=0.0, value=33.0)
        cost_v3 = st.number_input("Cost of V3", min_value=0.0, value=29.0536)
        v1_capacity = st.number_input("Capacity of V1", min_value=1, value=64)
        v2_capacity = st.number_input("Capacity of V2", min_value=1, value=66)
        v3_capacity = st.number_input("Capacity of V3", min_value=1, value=72)
        scenario = st.selectbox("Select Scenario", ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"])
        
        if st.button("Optimize Load"):
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
            
            st.session_state['vehicle_assignments'] = vehicle_assignments
            st.session_state['df_locations'] = df_locations
            
        if 'vehicle_assignments' in st.session_state:
            st.write("### Route Generation")
            if st.button("Generate Routes"):
                vehicle_assignments = st.session_state['vehicle_assignments']
                df_locations = st.session_state['df_locations']
                
                vehicle_routes = generate_routes(vehicle_assignments, df_locations)
                summary_df = display_summary(vehicle_routes)
                
                generate_excel(vehicle_routes, summary_df)
