import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
import base64
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

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
# Local Background Image Setup
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
        color: #E0FFFF;
        font-size: 40px;
        text-align: center;
        margin-bottom: 10px;
    }}
    .subtitle {{
        font-family: 'Helvetica Neue', sans-serif;
        color: #FFFF99;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .recommendation-card {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Styling
# ---------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(120deg, white, white);
        color: white;
        font-family: 'Helvetica Neue', sans-serif;
    }
    [data-testid="stSidebar"] .css-1d391kg {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Data Loading and ML Model
# ---------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("destinations_with_coordinates.csv")
    
    # Clean and normalize data
    data.columns = data.columns.str.strip().str.lower()
    data["pname"] = data["pname"].str.strip().str.lower()
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    
    # Process tags
    data['tags'] = data['tags'].apply(
        lambda x: ','.join([tag.strip().lower() for tag in str(x).split(',')]) 
        if pd.notna(x) else np.nan
    )
    
    # Feature engineering
    feature_cols = ['culture', 'adventure', 'wildlife', 'sightseeing', 'history']
    scaler = MinMaxScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    # Train ML model
    data['popularity'] = data[feature_cols].mean(axis=1) + 0.1*data['culture']
    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_cols], data['popularity'], test_size=0.2, random_state=42
    )
    
    lgbm_model = lgbm.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100)
    lgbm_model.fit(X_train, y_train)
    data['ml_score'] = lgbm_model.predict(data[feature_cols])
    
    # Normalize ML scores
    data['ml_score'] = MinMaxScaler().fit_transform(data[['ml_score']])
    
    return data, scaler, feature_cols, lgbm_model

data, scaler, feature_cols, lgbm_model = load_data()

# ---------------------------
# Weather API
# ---------------------------
def get_weather(lat, lon, api_key="13e12c96cea3ae958db69de1f4bf41bd"):
    try:
        if pd.isna(lat) or pd.isna(lon):
            return "No weather data"
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        description = weather_data['weather'][0]['description'].capitalize()
        temp = weather_data['main']['temp']
        return f"{description}, {temp}°C"
    except Exception as e:
        return "Weather unavailable"

# ---------------------------
# Recommendation Engine
# ---------------------------
def recommend_destinations(user_preferences, input_destination, selected_tags, data, top_n=6):
    similarity_scores = np.zeros(len(data))
    input_destination = input_destination.strip().lower() if input_destination else ""
    
    # Destination-based similarity
    if input_destination:
        if input_destination in data['pname'].values:
            destination_idx = data[data['pname'] == input_destination].index[0]
            destination_features = data.loc[destination_idx, feature_cols].values.reshape(1, -1)
            similarity_scores += cosine_similarity(data[feature_cols].values, destination_features).flatten()
        else:
            st.warning(f"Destination '{input_destination}' not found")
    
    # User preferences similarity
    if user_preferences:
        user_df = pd.DataFrame([user_preferences])
        user_scaled = scaler.transform(user_df)
        similarity_scores += cosine_similarity(data[feature_cols].values, user_scaled).flatten()
    
    # Tag-based matching
    if selected_tags:
        normalized_tags = [tag.strip().lower() for tag in selected_tags]
        max_tag_score = len(normalized_tags)
        for idx, row in data.iterrows():
            if pd.notna(row['tags']):
                row_tags = [t.strip().lower() for t in row['tags'].split(',')]
                tag_score = sum(1 for tag in normalized_tags if tag in row_tags)
                similarity_scores[idx] += (tag_score / max_tag_score) if max_tag_score > 0 else 0
    
    # Blend with ML predictions
    similarity_scores = 0.7 * similarity_scores + 0.3 * data['ml_score']
    
    # Normalize final scores
    similarity_scores = (similarity_scores - np.min(similarity_scores)) / (
        np.max(similarity_scores) - np.min(similarity_scores) + 1e-8
    )
    
    data_copy = data.copy()
    data_copy['similarity'] = similarity_scores
    
    if input_destination:
        data_copy = data_copy[data_copy['pname'] != input_destination]
    
    sorted_data = data_copy.sort_values(by='similarity', ascending=False)
    
    recommendations = []
    for _, row in sorted_data.tail(top_n).iterrows():
        lat, lon = row['latitude'], row['longitude']
        weather_info = get_weather(lat, lon)
        province_val = row.get('province', "")
        if province_val and not str(province_val).lower().startswith("province"):
            province_val = f"Province {province_val}"
        recommendations.append({
            "Destination": row['pname'].title(),
            "Similarity": round(row['similarity'], 3),
            "Tags": row['tags'].title() if pd.notna(row['tags']) else "",
            "Weather": weather_info,
            "Province": province_val
        })
    return pd.DataFrame(recommendations)

# ---------------------------
# UI Components
# ---------------------------
def generate_recommendations():
    dest = st.session_state.input_destination.strip().lower() if st.session_state.input_destination else ""
    prefs = st.session_state.get("user_preferences", None)
    tags = [tag.strip().lower() for tag in st.session_state.get("selected_tags", [])]
    recs = recommend_destinations(prefs, dest, tags, data)
    st.session_state.recommendations = recs

# Sidebar
st.sidebar.header("Customize Recommendations")
st.sidebar.text_input("Search destination:", key="input_destination", 
                     on_change=generate_recommendations)

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

add_tags = st.sidebar.checkbox("Add tags?")
if add_tags:
    all_tags = sorted(list(set(
        tag.strip().lower()
        for tags in data['tags'].dropna()
        for tag in tags.split(',')
    )))
    st.session_state.selected_tags = st.sidebar.multiselect(
        "Select tags:", 
        [tag.title() for tag in all_tags]
    )

st.sidebar.button("✨ Get Recommendations", on_click=generate_recommendations)

# Main UI
st.markdown("<div class='title'>🌍 Destination Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find your next adventure!</div>", unsafe_allow_html=True)

if "recommendations" in st.session_state and st.session_state.recommendations is not None:
    recs = st.session_state.recommendations
    if not recs.empty:
        st.subheader("Top Recommendations")
        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""<div class="recommendation-card">
                    <strong>{row['Destination']}</strong><br>
                    {"📍 " + row['Province'] if row.get("Province") else ""}<br>
                    <strong>Match Score:</strong> {row['Similarity']}<br>
                    <strong>Tags:</strong> {row['Tags']}<br>
                    <strong>Weather:</strong> {row['Weather']}
                </div>""", unsafe_allow_html=True)
    else:
        st.warning("No recommendations found. Try different inputs.")
else:
    st.info("Enter a destination or set preferences to get recommendations")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** Combine different filters for better results!")