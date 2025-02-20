import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
import base64

# ---------------------------
# Configuration & Setup
# ---------------------------
st.set_page_config(
    page_title="🌍 Destination Recommender",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Local Background Image Setup for Main App
# ---------------------------
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

base64_nepal = get_base64_image("nepal.png")
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_nepal}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .title {{
        font-family: 'Helvetica Neue', sans-serif;
        color: #444;
        font-size: 40px;
        text-align: center;
        margin-bottom: 10px;
    }}
    .subtitle {{
        font-family: 'Helvetica Neue', sans-serif;
        color: grey;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }}
    /* Recommendation card with a solid light background for clarity */
    .recommendation-card {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Attractive Sidebar Styling
# ---------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(120deg, blue, white);
        color: white;
        font-family: 'Helvetica Neue', sans-serif;
    }
    [data-testid="stSidebar"] .css-1d391kg {
        padding: 1rem;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
@st.cache_data
def load_data():
    # Read CSV with default comma delimiter
    data = pd.read_csv("destinations_with_coordinates.csv")
    # Convert column names to lowercase and strip whitespace
    data.columns = data.columns.str.strip().str.lower()
    
    # Ensure latitude and longitude are numeric
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    
    # Define feature columns and scale them
    feature_cols = ['culture', 'adventure', 'wildlife', 'sightseeing', 'history']
    scaler = MinMaxScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data, scaler, feature_cols

data, scaler, feature_cols = load_data()

# ---------------------------
# Weather API Function (Working Example)
# ---------------------------
def get_weather(lat, lon, api_key="13e12c96cea3ae958db69de1f4bf41bd"):
    try:
        if pd.isna(lat) or pd.isna(lon):
            return "No weather data available"
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        description = weather_data['weather'][0]['description'].capitalize()
        temp = weather_data['main']['temp']
        return f"{description}, {temp}°C"
    except Exception as e:
        return "Weather data unavailable"

# ---------------------------
# Recommendation Engine
# ---------------------------
def recommend_destinations(user_preferences, input_destination, selected_tags, data, top_n=6):
    """
    Recommend destinations based on user preferences, destination name, and tags.
    """
    similarity_scores = np.zeros(len(data))
    data["pname"] = data["pname"].str.strip().str.lower()
    # Similarity based on destination name
    if input_destination:
        if input_destination in data['pname'].values:
            destination_idx = data[data['pname'] == input_destination].index[0]
            destination_features = data.loc[destination_idx, feature_cols].values.reshape(1, -1)
            similarity_scores += cosine_similarity(data[feature_cols].values, destination_features).flatten()
        else:
            st.warning(f"Destination '{input_destination}' not found in the dataset.")
    if input_destination:
        input_destination = input_destination.strip().lower()  # Normalize input
    
 
    # Similarity based on user preferences
    if user_preferences:
        user_df = pd.DataFrame([user_preferences])
        user_scaled = scaler.transform(user_df)
        similarity_scores += cosine_similarity(data[feature_cols].values, user_scaled).flatten()
    
    # Similarity based on tags
    if selected_tags:
        tag_weights = {tag: 1 for tag in selected_tags}
        for idx, row in data.iterrows():
            row_tags = row['tags'].split(',') if pd.notna(row['tags']) else []
            for tag in row_tags:
                if tag.strip() in tag_weights:
                    similarity_scores[idx] += tag_weights[tag.strip()]
    
    data_copy = data.copy()  # work on a copy to avoid modifying the original DataFrame
    data_copy['similarity'] = similarity_scores
    # Exclude the starting destination from recommendations
    if input_destination:
        data_copy = data_copy[data_copy['pname'] != input_destination]
    sorted_data = data_copy.sort_values(by='similarity', ascending=False)
    
    recommendations = []
    for _, row in sorted_data.head(top_n).iterrows():
        lat, lon = row['latitude'], row['longitude']
        weather_info = get_weather(lat, lon)
        province_val = row.get('province', "")
        if province_val and not str(province_val).lower().startswith("province"):
            province_val = f"Province {province_val}"
        recommendations.append({
            "Destination": row['pname'],
            "Similarity": round(row['similarity'], 3),
            "Tags": row['tags'],
            "Weather": weather_info,
            "Province": province_val
        })
    return pd.DataFrame(recommendations)

# ---------------------------
# Callback Function to Generate Recommendations on Enter
# ---------------------------
def generate_recommendations():
    dest = st.session_state.input_destination
    prefs = st.session_state.get("user_preferences", None)
    tags = st.session_state.get("selected_tags", None)
    recs = recommend_destinations(prefs, dest, tags, data, top_n=6)
    st.session_state.recommendations = recs

# ---------------------------
# Sidebar Inputs with Attractive Styling
# ---------------------------
st.sidebar.header("Customize Your Recommendations")

# Use on_change to generate recommendations when Enter is pressed
st.sidebar.text_input("Enter a destination name:", key="input_destination", on_change=generate_recommendations)

add_preferences = st.sidebar.checkbox("Set preferences?")
if add_preferences:
    st.sidebar.subheader("Your Preferences")
    st.session_state.user_preferences = {
        "culture": st.sidebar.slider("Culture", 0, 5, 3),
        "adventure": st.sidebar.slider("Adventure", 0, 5, 3),
        "wildlife": st.sidebar.slider("Wildlife", 0, 5, 3),
        "sightseeing": st.sidebar.slider("Sightseeing", 0, 5, 3),
        "history": st.sidebar.slider("History", 0, 5, 3)
    }
else:
    st.session_state.user_preferences = None

add_tags = st.sidebar.checkbox("Add tags?")
if add_tags:
    all_tags = list(set(tag.strip() for tags in data['tags'].dropna() for tag in tags.split(',')))
    st.session_state.selected_tags = st.sidebar.multiselect("Select tags:", all_tags)
else:
    st.session_state.selected_tags = None

if st.sidebar.button("✨ Generate Recommendations"):
    generate_recommendations()

# ---------------------------
# Main App Layout
# ---------------------------
st.markdown("<div class='title'>🌍 Destination Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find your next adventure, tailored to your preferences!</div>", unsafe_allow_html=True)

if "recommendations" in st.session_state and st.session_state.recommendations is not None:
    recs = st.session_state.recommendations
    if not recs.empty:
        st.subheader("Top Recommendations")
        # Display recommendations in a grid (3 columns per row)
        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""<div class="recommendation-card">
                <strong>{row['Destination']}</strong><br>
                {"📍 " + row['Province'] if row.get("Province", "") else ""}<br>
                <strong>Similarity:</strong> {row['Similarity']}<br>
                <strong>Tags:</strong> {row['Tags']}<br>
                <strong>Weather:</strong> {row['Weather']}<br>
                </div>""", unsafe_allow_html=True)
    else:
        st.warning("No recommendations found. Try adjusting your inputs.")
else:
    st.info("Enter a destination (and optionally set preferences or tags) and press Enter to get recommendations.")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** Combine destination name, preferences, and tags for personalized results.")
