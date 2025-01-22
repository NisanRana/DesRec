# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set a random seed for reproducibility
RANDOM_SEED = 42

# Load and preprocess the dataset
data = pd.read_csv("destinations_with_coordinates.csv")

# Scale features for clustering
scaler = MinMaxScaler()
data[['culture', 'adventure', 'wildlife', 'sightseeing', 'history']] = scaler.fit_transform(
    data[['culture', 'adventure', 'wildlife', 'sightseeing', 'history']]
)

# Define the feature set for clustering
features = data[['culture', 'adventure', 'wildlife', 'sightseeing', 'history']]

# Perform KMeans clustering
optimal_k = 5  # Adjust based on your elbow method results
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED)
data['cluster'] = kmeans.fit_predict(features)

# Function to recommend destinations based on user preferences, destination name, and tags
def recommend_destinations(user_preferences, input_destination, selected_tags, data, top_n=5):
    """
    Recommend destinations based on user preferences, input destination, and selected tags.

    Parameters:
        user_preferences (dict): User's feature preferences.
        input_destination (str): Destination name provided by the user.
        selected_tags (list): Tags to refine recommendations.
        data (pd.DataFrame): Dataset with destination features and cluster labels.
        top_n (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: Top N recommended destinations.
    """
    # Scale user preferences
    user_df = None
    if user_preferences:
        user_df = pd.DataFrame([user_preferences])
        user_df = scaler.transform(user_df)

    # Handle input destination similarity
    similarity_scores = np.zeros(len(data))
    if input_destination:
        if input_destination in data['pName'].values:
            destination_idx = data[data['pName'] == input_destination].index[0]
            # Access the feature matrix as a NumPy array
            destination_features = features.iloc[destination_idx].values.reshape(1, -1)
            similarity_scores += cosine_similarity(features.values, destination_features).flatten()
        else:
            st.warning(f"Destination '{input_destination}' not found in the dataset.")

    # Handle user preferences similarity
    if user_preferences:
        similarity_scores += cosine_similarity(features.values, user_df).flatten()

    # Add tag-based weights
    if selected_tags:
        tag_weights = {tag: 1 for tag in selected_tags}
        for idx, row in data.iterrows():
            row_tags = row['tags'].split(',') if pd.notna(row['tags']) else []
            for tag in row_tags:
                if tag in tag_weights:
                    similarity_scores[idx] += tag_weights[tag]

    # Combine scores and sort
    data['similarity'] = similarity_scores

    # Exclude the input destination from the results
    sorted_data = data[data['pName'] != input_destination].sort_values(by='similarity', ascending=False)

    # Return the top N recommendations
    recommendations = sorted_data.head(top_n)[['pName', 'similarity', 'tags']]
    return recommendations


# Streamlit UI
st.title("Destination Recommender")

st.write("""
Welcome to the Destination Recommender App! 
You can search for recommendations by destination name, user preferences, or a combination of both.
Additionally, you can add tags to refine your search.
""")

# User choice for recommendation method
st.subheader("How would you like to search for recommendations?")
search_by_name = st.checkbox("Search by Destination Name")
search_by_preferences = st.checkbox("Search by User Preferences")
add_tags = st.checkbox("Add Tags to Refine Recommendations")

# Input destination name
input_destination = None
if search_by_name:
    input_destination = st.text_input("Enter the destination name:", "")

# Input user preferences
user_preferences = None
if search_by_preferences:
    st.subheader("Set Your Preferences")
    user_preferences = {
        "culture": st.slider("Culture", 0.0, 1.0, 0.5),
        "adventure": st.slider("Adventure", 0.0, 1.0, 0.5),
        "wildlife": st.slider("Wildlife", 0.0, 1.0, 0.5),
        "sightseeing": st.slider("Sightseeing", 0.0, 1.0, 0.5),
        "history": st.slider("History", 0.0, 1.0, 0.5),
    }

# Input tags
selected_tags = None
if add_tags:
    all_tags = list(set(tag.strip() for tags in data['tags'].dropna() for tag in tags.split(',')))
    selected_tags = st.multiselect("Select tags:", all_tags)

# Recommendation button
if st.button("Get Recommendations"):
    if not search_by_name and not search_by_preferences:
        st.warning("Please select at least one method: Search by Destination Name or User Preferences.")
    else:
        recommendations = recommend_destinations(user_preferences, input_destination, selected_tags, data, top_n=5)

        if not recommendations.empty:
            st.subheader("Top Recommendations")
            st.table(recommendations)
        else:
            st.warning("No recommendations found. Please adjust your inputs.")
