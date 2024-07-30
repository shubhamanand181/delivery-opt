import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp

# Streamlit setup
st.title("Delivery Cost and Route Optimization")

# File uploader for Excel file
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Instructions for Excel Upload
st.write("""
### Instructions:
1. The Excel sheet must have column names in the first row.
2. The sheet to be analyzed must contain columns named "Party", "Latitude", "Longitude", and "Weight (KG)".
""")

# Function to calculate distance matrix
def calculate_distance_matrix(df):
    num_locations = len(df)
    distance_matrix = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                try:
                    coords_1 = (float(df.iloc[i]['Latitude']), float(df.iloc[j]['Longitude']))
                    coords_2 = (float(df.iloc[j]['Latitude']), float(df.iloc[j]['Longitude']))
                    distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                except ValueError as e:
                    st.write(f"Invalid coordinates at index {i} or {j}: {e}")
                    distance_matrix[i][j] = np.inf  # Assign a large value to indicate invalid distance
            else:
                distance_matrix[i][j] = 0
    return distance_matrix

# Function to optimize load
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

# Function to assign deliveries to vehicles based on load optimization result
def assign_deliveries_to_vehicles(df, load_result, v1_capacity, v2_capacity, v3_capacity):
    vehicle_assignments = {'V1': [], 'V2': [], 'V3': []}

    deliveries = df.sort_values(by='Weight (KG)', ascending=False)

    # Function to assign deliveries to a vehicle
    def assign_to_vehicle(vehicle, capacity):
        assigned_deliveries = []
        remaining_capacity = capacity
        for i, delivery in deliveries.iterrows():
            if remaining_capacity > 0:
                assigned_deliveries.append(delivery)
                remaining_capacity -= 1
            else:
                break
        return assigned_deliveries

    # Assign deliveries to V1
    vehicle_assignments['V1'] = assign_to_vehicle('V1', int(load_result['Deliveries assigned to V1']))
    deliveries = deliveries[~deliveries.index.isin([x.name for x in vehicle_assignments['V1']])]

    # Assign deliveries to V2
    vehicle_assignments['V2'] = assign_to_vehicle('V2', int(load_result['Deliveries assigned to V2']))
    deliveries = deliveries[~deliveries.index.isin([x.name for x in vehicle_assignments['V2']])]

    # Assign deliveries to V3
    vehicle_assignments['V3'] = assign_to_vehicle('V3', int(load_result['Deliveries assigned to V3']))
    deliveries = deliveries[~deliveries.index.isin([x.name for x in vehicle_assignments['V3']])]

    return vehicle_assignments

# Function to optimize route for each vehicle's assigned deliveries
def optimize_vehicle_route(vehicle, assigned_deliveries, df):
    if len(assigned_deliveries) == 0:
        return [], 0
    cluster = df[df['Party'].isin([delivery['Party'] for delivery in assigned_deliveries])]
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

    # Optimize route for the vehicle
    route, total_distance = nearest_neighbor(distance_matrix)
    mapped_route = cluster.index[route]

    route_details = df.loc[mapped_route].to_dict('records')
    return route_details, total_distance

# Function to generate an Excel file for route details
def generate_excel(vehicle_routes):
    writer = pd.ExcelWriter('Vehicle_Routes.xlsx', engine='xlsxwriter')

    for vehicle, route in vehicle_routes.items():
        df = pd.DataFrame(route)
        df.to_excel(writer, sheet_name=vehicle, index=False)

    writer.save()

# Function to optimize routes within each cluster
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

if uploaded_file:
    df_locations = pd.read_excel(uploaded_file)

    # Ensure column names are as expected
    expected_columns = ['Party', 'Latitude', 'Longitude', 'Weight (KG)']
    if all(col in df_locations.columns for col in expected_columns):
        st.write("All expected columns are present.")
    else:
        st.write("One or more expected columns are missing. Please check the column names in the Excel file.")

    # Remove rows with NaN values in Latitude or Longitude
    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # Example Load Optimization with extracted data
    D_a = sum((df_locations['Weight (KG)'] > 0) & (df_locations['Weight (KG)'] <= 2))
    D_b = sum((df_locations['Weight (KG)'] > 2) & (df_locations['Weight (KG)'] <= 10))
    D_c = sum((df_locations['Weight (KG)'] > 10) & (df_locations['Weight (KG)'] <= 200))

    # Define vehicle parameters
    cost_v1 = 62.8156
    cost_v2 = 33.0
    cost_v3 = 29.0536
    v1_capacity = 64
    v2_capacity = 66
    v3_capacity = 72
    scenario = "Scenario 1: V1, V2, V3"

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

    # Assign deliveries to vehicles
    vehicle_assignments = assign_deliveries_to_vehicles(df_locations, result, v1_capacity, v2_capacity, v3_capacity)
    st.write("Vehicle Assignments:")
    st.write(f"V1: {len(vehicle_assignments['V1'])} deliveries")
    st.write(f"V2: {len(vehicle_assignments['V2'])} deliveries")
    st.write(f"V3: {len(vehicle_assignments['V3'])} deliveries")

    # Optimize route for each vehicle
    vehicle_routes = {}
    total_distances = {}
    for vehicle, assignments in vehicle_assignments.items():
        route_details, total_distance = optimize_vehicle_route(vehicle, assignments, df_locations)
        vehicle_routes[vehicle] = route_details
        total_distances[vehicle] = total_distance

    # Display route details
    for vehicle, route in vehicle_routes.items():
        st.write(f"\n{vehicle} Route:")
        st.write(pd.DataFrame(route))

    # Generate Excel file with route details
    generate_excel(vehicle_routes)
    st.write("Excel file with route details generated.")

    # Print the total distances for each vehicle
    for vehicle, distance in total_distances.items():
        st.write(f"{vehicle} Total Distance: {distance / 1000:.2f} kilometers")

