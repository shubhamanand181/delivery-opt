import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp

# Load data from the uploaded file
st.title("Delivery Optimization and Route Planning")

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file:
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
        st.stop()

    # Remove rows with NaN values in Latitude or Longitude
    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    def calculate_distance_matrix(df):
        num_locations = len(df)
        distance_matrix = np.zeros((num_locations, num_locations))

        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    try:
                        coords_1 = (float(df.loc[i, 'Latitude']), float(df.loc[i, 'Longitude']))
                        coords_2 = (float(df.loc[j, 'Latitude']), float(df.loc[j, 'Longitude']))
                        distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                    except ValueError as e:
                        st.write(f"Invalid coordinates at index {i} or {j}: {e}")
                        distance_matrix[i][j] = np.inf  # Assign a large value to indicate invalid distance
                else:
                    distance_matrix[i][j] = 0
        return distance_matrix

    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(df_locations)

    # Define the maximum distance for points to be considered in the same cluster
    epsilon = 200  # meters

    # Apply DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
    db.fit(distance_matrix)

    # Add cluster labels to the DataFrame
    df_locations['Cluster'] = db.labels_

    # Calculate centroids of clusters
    centroids = df_locations.groupby('Cluster')[['Latitude', 'Longitude']].mean()

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

    # Calculate distance matrix for each cluster and plan routes
    vehicle_routes = {"V1": [], "V2": [], "V3": []}
    clusters = df_locations['Cluster'].unique()

    summary_data = []

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

        vehicle_name = f"V1" if cluster_id % 3 == 0 else f"V2" if cluster_id % 3 == 1 else f"V3"
        vehicle_routes[vehicle_name].append({
            "Cluster": cluster_id,
            "Route": df_locations.loc[mapped_route, expected_columns].to_dict('records'),
            "Total Distance (km)": total_distance / 1000  # Scale down for display
        })

        summary_data.append({
            "Cluster ID": cluster_id,
            "Vehicle": vehicle_name,
            "Number of Shops": num_locations,
            "Centroid Latitude": centroids.loc[cluster_id, 'Latitude'],
            "Centroid Longitude": centroids.loc[cluster_id, 'Longitude'],
            "Total Distance (km)": total_distance / 1000
        })

    summary_df = pd.DataFrame(summary_data)

    # Display updated centroids section with additional columns
    st.subheader("Centroids Information")
    st.write(summary_df)

    # Function to generate Excel file
    def generate_excel(vehicle_routes, summary_df):
        with pd.ExcelWriter("optimized_routes.xlsx", engine="xlsxwriter") as writer:
            # Write summary sheet
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            for vehicle, routes in vehicle_routes.items():
                for idx, route_info in enumerate(routes):
                    route_df = pd.DataFrame(route_info["Route"])
                    route_df['Sequence'] = range(1, len(route_df) + 1)
                    route_df.to_excel(writer, sheet_name=f"{vehicle}_Cluster_{route_info['Cluster']}", index=False)
            writer.close()

    generate_excel(vehicle_routes, summary_df)

    st.write("Excel file with optimized routes for each vehicle and summary:")
    st.download_button(
        label="Download Excel file",
        data=open("optimized_routes.xlsx", "rb").read(),
        file_name="optimized_routes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
