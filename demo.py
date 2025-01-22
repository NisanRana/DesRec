import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Set random seed
RANDOM_SEED = 42

# Load and preprocess the dataset
data = pd.read_csv("destinations_with_coordinates.csv")

# Scale features for clustering
scaler = MinMaxScaler()
data[['culture', 'adventure', 'wildlife', 'sightseeing', 'history']] = scaler.fit_transform(
    data[['culture', 'adventure', 'wildlife', 'sightseeing', 'history']]
)

# Feature matrix
features = data[['culture', 'adventure', 'wildlife', 'sightseeing', 'history']]

##API INTEGRATION
import requests

# Function to get weather data
def get_weather(lat, lon, api_key="13e12c96cea3ae958db69de1f4bf41bd"):

    try:
        # Check for valid coordinates
        if pd.isna(lat) or pd.isna(lon):
            return "No weather data available"
        
        # OpenWeatherMap API URL
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()

        # Parse API response
        weather_data = response.json()
        description = weather_data['weather'][0]['description'].capitalize()
        temp = weather_data['main']['temp']
        return f"{description}, {temp}°C"
    except Exception as e:
        return "Weather data unavailable"



def recommend_destinations(user_preferences, input_destination, selected_tags, data, top_n=5):
    """
    Recommend destinations based on user preferences, destination name, and tags.
    Includes weather information.
    """
    similarity_scores = np.zeros(len(data))

    # Add similarity by destination name
    if input_destination:
        if input_destination in data['pName'].values:
            destination_idx = data[data['pName'] == input_destination].index[0]
            destination_features = features.iloc[destination_idx].values.reshape(1, -1)
            similarity_scores += cosine_similarity(features.values, destination_features).flatten()
        else:
            st.warning(f"Destination '{input_destination}' not found in the dataset.")

    # Add similarity by user preferences
    if user_preferences:
        user_df = pd.DataFrame([user_preferences])
        user_df = scaler.transform(user_df)
        similarity_scores += cosine_similarity(features.values, user_df).flatten()

    # Add similarity by tags
    if selected_tags:
        tag_weights = {tag: 1 for tag in selected_tags}
        for idx, row in data.iterrows():
            row_tags = row['tags'].split(',') if pd.notna(row['tags']) else []
            for tag in row_tags:
                if tag in tag_weights:
                    similarity_scores[idx] += tag_weights[tag]

    # Calculate final similarity scores
    data['similarity'] = similarity_scores
    sorted_data = data[data['pName'] != input_destination].sort_values(by='similarity', ascending=False)
    
    # Prepare recommendations
    recommendations = []
    for _, row in sorted_data.head(top_n).iterrows():
        lat, lon = row['latitude'], row['longitude']
        weather_info = get_weather(lat, lon)  # Fetch weather data using the coordinates
        recommendations.append({
            "Destination": row['pName'],
            "Similarity": round(row['similarity'], 3),
            "Tags": row['tags'],
            "Weather": weather_info
        })

    return pd.DataFrame(recommendations)


    # Append weather info to recommendations
    for _, row in sorted_data.head(top_n).iterrows():
        lat, lon = row.get('latitude'), row.get('longitude')
        weather_info = get_weather(lat, lon, api_key)
        recommendations.append({
            "Destination": row['pName'],
            "Similarity": round(row['similarity'], 3),
            "Tags": row['tags'],
            "Weather": weather_info
        })

    return pd.DataFrame(recommendations)


# Streamlit UI Enhancements
st.set_page_config(page_title="Destination Recommender", layout="wide", page_icon="🌍")

st.markdown(
    """
    <style>
        .stApp {
            background-image: linear-gradient(to bottom, #ffecd2, #fcb69f);
            color: #3c3c3c;
        }
        .title {
            font-family: 'Helvetica Neue', sans-serif;
            color: #444;
            font-size: 40px;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-family: 'Helvetica Neue', sans-serif;
            color: #666;
            font-size: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>🌍 Destination Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find your next adventure, tailored to your preferences!</div>", unsafe_allow_html=True)

# Input section
st.sidebar.header("Customize Your Recommendations")
input_destination = st.sidebar.text_input("Enter a destination name:")

add_preferences = st.sidebar.checkbox("Set preferences?")
user_preferences = None
if add_preferences:
    st.sidebar.subheader("Your Preferences")
    user_preferences = {
        "culture": st.sidebar.slider("Culture", 0, 5, 3),
        "adventure": st.sidebar.slider("Adventure", 0, 5, 3),
        "wildlife": st.sidebar.slider("Wildlife", 0, 5, 3),
        "sightseeing": st.sidebar.slider("Sightseeing", 0, 5, 3),
        "history": st.sidebar.slider("History", 0, 5, 3),
    }

add_tags = st.sidebar.checkbox("Add tags?")
selected_tags = None
if add_tags:
    all_tags = list(set(tag.strip() for tags in data['tags'].dropna() for tag in tags.split(',')))
    selected_tags = st.sidebar.multiselect("Select tags:", all_tags)

if input_destination or user_preferences:
    recommendations = recommend_destinations(user_preferences, input_destination, selected_tags, data, top_n=5)
    if not recommendations.empty:
        st.subheader("Top Recommendations")
        st.table(recommendations)
    else:
        st.warning("No recommendations found. Try adjusting your inputs.")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip**: Combine destination name, preferences, and tags for personalized results.")
