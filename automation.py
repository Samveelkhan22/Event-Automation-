import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.impute import SimpleImputer
import sqlalchemy
from sklearn.metrics import pairwise_distances

# Create the database connection URL
db_url = "mysql+pymysql://root:@localhost/automation"

# Create the engine and establish the connection
engine = sqlalchemy.create_engine(db_url)
connection = engine.connect()

# Check if the connection is successful
if connection:
    print("Connection to the database is successful.")

# Load the data from the tables
users = pd.read_sql("SELECT DISTINCT * FROM users", connection)
venues = pd.read_sql("SELECT * FROM venues", connection)
events = pd.read_sql("SELECT * FROM events", connection)
user_event_invites = pd.read_sql("SELECT * FROM user_event_invites", connection)
users_want_to_meet = pd.read_sql("SELECT * FROM users_want_to_meet", connection)

# Group users by city, country
grouped_users = users.groupby(["city", "country"])

def perform_spectral_clustering(group, users_want_to_meet):
    # Get the unique user IDs from the group
    unique_user_ids = group["user_id"].unique()

    # Create an empty list to store the cluster labels
    cluster_labels = []

    for user_id in unique_user_ids:
        if user_id not in users_want_to_meet["from_user_id"].unique():
            cluster_labels.append(-1)  # No cluster assigned
        else:
            # Filter the users_want_to_meet DataFrame for the specific user_id
            user_data = users_want_to_meet[users_want_to_meet["from_user_id"] == user_id]

            # Get the target user IDs for clustering
            target_user_ids = user_data["to_user_id"].unique()

            # Calculate similarity matrix
            similarity_matrix = pd.DataFrame(0, index=unique_user_ids, columns=unique_user_ids)

            for from_user_id in unique_user_ids:
                if from_user_id == user_id:
                    similarity_matrix.loc[from_user_id, :] = 1
                elif from_user_id in target_user_ids:
                    similarity_matrix.loc[from_user_id, user_id] = 1
                    similarity_matrix.loc[user_id, from_user_id] = 1

            # Perform spectral clustering
            similarity_matrix_np = similarity_matrix.to_numpy()
            dissimilarity_matrix = 1 - similarity_matrix_np
            embedding = pairwise_distances(dissimilarity_matrix, metric="euclidean")
            cluster_model = SpectralClustering(n_clusters=2, affinity='precomputed')  # Adjust the number of clusters as needed
            cluster_label = cluster_model.fit_predict(embedding)[0]  # Assume user_id is the first in the embedding

            cluster_labels.append(cluster_label)

    # Ensure that unique_user_ids and cluster_labels have the same length
    unique_user_ids = unique_user_ids[:len(cluster_labels)]

    # Create the cluster_data DataFrame
    cluster_data = pd.DataFrame({"user_id": unique_user_ids, "cluster_label": cluster_labels})

    return cluster_data

# Iterate over each group and perform clustering and event creation
for _, group in grouped_users:
    # Apply spectral clustering on mutual matches within the group
    grouped_users_clustered = perform_spectral_clustering(group, users_want_to_meet)

    # Iterate over each cluster within the group
    clusters = grouped_users_clustered.groupby("cluster_label")
    for _, cluster in clusters:
        # Create a unique event for each cluster
        event_id = np.random.randint(1, 10000)
        event_title = "Mutually Matched Group Event"
        event_start_time = np.random.choice(["Friday", "Saturday", "Sunday"], p=[0.7, 0.2, 0.1])
        event_end_time = event_start_time + " 21:00"
        venue_id = np.random.choice(venues["venue_id"])
        event_data = events.loc[events["event_id"] == event_id, "event_photo"]
        event_image = event_data.iloc[0] if not event_data.empty else None



        # Insert the event into the events table
        event_data = {
            "event_id": event_id,
            "event_title": event_title,
            "start_time": event_start_time,
            "end_time": event_end_time,
            "venue_id": venue_id,
            "event_image": event_image
        }
        event_data_df = pd.DataFrame([event_data])
        events = pd.concat([events, event_data_df], ignore_index=True)


        # Assign users to the event by inserting records into the event_invites table
        users_in_cluster = cluster["user_id"].tolist()
        user_event_invites_data = {
            "user_id": users_in_cluster,
            "event_id": [event_id] * len(users_in_cluster)
        }
        user_event_invites = pd.concat([user_event_invites, pd.DataFrame(user_event_invites_data)], ignore_index=True)

# Save the updated tables to CSV files
events.to_csv("events_output.csv", index=False)
user_event_invites.to_csv("user_event_invites_output.csv", index=False)

# Save the selected tables to CSV files
users.to_csv("users_table.csv", index=False)
venues.to_csv("venues_table.csv", index=False)

# Close the connection to the database
connection.close()
