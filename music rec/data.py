import streamlit as st
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Set Streamlit theme and layout
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and preprocess the dataset
data = pd.read_csv('Spotify.csv')
data= data.iloc[0:999]

genre_dict = {
    "dance pop" : "pop", 
    "pop soul" : "soul",
    "atl hip hop" : "hip hop",
    "pop rap" : "rap",
    "big room" : "house",
    "canadian hip hop" : "hip hop",
    "disco house" : "house",
    "romanian house" : "house",
    'lilith' : "indie",
    'detroit hip hop' : "hip hop", 
    'asian american hip hop' : "hip hop", 
    'east coast hip hop': "hip hop",
    'neo mellow' : "pop", 
    'canadian pop' : "pop", 
    'reggae fusion' : "reggae", 
    'idol' : "pop", 
    'art pop' : "pop",
    "talent show" : "pop", 
    'modern alternative rock' : "rock",
    'indietronica' : "electronic", 
    'grime' : "hip hop", 
    'barbadian pop' : "pop", 
    'acoustic pop' : "pop",
    'dutch house' : "house", 
    'belgian pop' : "pop", 
    'contemporary country' : "country", 
    'boy band' : "pop",
    'celtic rock' : "rock", 
    'edm' : "electronic", 
    'indie rock' : "indie", 
    'australian dance' : "dance",
    'british soul' : "soul", 
    'eau claire indie' : "indie", 
    'dancefloor dnb' : "dance",
    'permanent wave' : "rock", 
    'hip pop' : "pop", 
    'g funk' : "funk", 
    'baroque pop' : "pop", 
    'indie pop' : "pop",
    'chicago rap' : "rap", 
    'indie poptimism' : "indie", 
    'french shoegaze' : "rock",
    'alternative metal' : "metal", 
    'indie folk' : "indie", 
    'alternative rock' : "rock",
    'uk hip hop' : "hip hop", 
    'electro house' : "house", 
    'garage rock' : "rock", 
    'israeli pop' : "pop",
    'alternative r&b' : "r&b", 
    'australian pop' : "pop", 
    'candy pop' : "pop", 
    'modern rock' : "rock",
    'conscious hip hop' : "hip hop", 
    'folk-pop' : "pop", 
    'alternative dance' : "dance", 
    'k-pop' : "pop",
    'gangster rap' : "rap", 
    'brostep' : "dance", 
    'downtempo' : "pop", 
    'la indie' : "indie", 
    'bass trap' : "dance",
    'metropopolis' : "pop", 
    'electropop' : "pop", 
    'electro' : "electronica", 
    'destroy techno' : "dance", 
    'emo' : "rock",
    'austrian pop' : "pop", 
    'irish pop' : "pop", 
    'adult standards' : "pop", 
    'modern folk rock' : "rock",
    'tropical house' : "house", 
    'contemporary r&b' : "r&b", 
    'deep disco house' : "house",
    'bubblegum dance' : "dance", 
    'chill pop' : "pop", 
    'comic' : "pop", 
    'complextro' : "electronica", 
    'nyc rap' : "rap",
    'deep groove house' : "house", 
    'australian hip hop' : "hip hop", 
    'neo soul' : "soul",
    'deep house' : "house", 
    'french indie pop' : "pop", 
    'german pop' : "pop", 
    'dutch hip hop' : "hip hop",
    'aussietronica' : "electronica", 
    'australian indie' : "indie", 
    'canadian contemporary r&b' : "r&b",
    'kentucky hip hop' : "hip hop", 
    'new jersey rap' : "rap", 
    'irish singer-songwriter' : "pop",
    'ghanaian hip hop' : "hip hop", 
    'icelandic indie' : "indie", 
    'indie pop rap' : "pop",
    'new french touch' : "pop", 
    'san diego rap' : "rap", 
    'australian psych' : "rock",
    'canadian indie' : "indie", 
    'alt z' : "pop", 
    'danish pop' : "pop", 
    'melodic rap' : "rap",
    'social media pop' : "pop", 
    'london rap' : "rap", 
    'florida rap' : "rap", 
    'emo rap' : "rap",
    'latin' : "dance", 
    'ohio hip hop' : "hip hop", 
    'dfw rap' : "rap", 
    'hawaiian hip hop' : "hio hop",
    'dirty south rap' : "rap", 
    'afroswing' : "swing", 
    'basshall' : "dance", 
    'memphis hip hop' : "hip hop",
    'bedroom pop' : "pop", 
    'hollywood' : "pop", 
    'afrofuturism' : "pop", 
    'comedy rap' : "rap",
    'colombian pop' : "pop", 
    'cali rap' : "rap", 
    'black americana' : "pop",
    'north carolina hip hop' : "hip hop", 
    'alternative pop rock' : "pop", 
    'dark clubbing' : "dance",
    'lgbtq+ hip hop' : "hip hop", 
    'afro dancehall' : "dance", 
    'argentine hip hop' : "hip hop",
    'classic rock' : "rock", 
    'uk drill' : "hip hop"}

for key, value in genre_dict.items():
    data["top genre"].replace(key, value, inplace = True)

data=data.drop_duplicates(subset="title")


# Drop rows with invalid 'duration_ms' values (non-numeric)
data = data[pd.to_numeric(data['dur'], errors='coerce').notna()]

# Extract numerical features (excluding 'time_signature')
numerical_features = ['acous', 'dnce', 'nrgy', 'pop', 'live', 'dB',
                      'spch', 'bpm', 'val']

# Extract categorical features and perform one-hot encoding
categorical_features = ['title']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(data[numerical_features])

# Streamlit web app
st.title("🎵 Music Recommendation System")

# Sidebar with user input
st.sidebar.header("User Input")
song_name = st.sidebar.text_input("Enter a song name:", "i need a dollar")
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=10)


# Recommendation function by song name
def recommend_tracks_by_name(song_name, num_recommendations):
    # Filter tracks that contain the given song name in their title
    matching_tracks = data[data['title'].str.contains(song_name, case=False, na=False)]

    if matching_tracks.empty:
        st.error(f"No tracks found with the name '{song_name}' in the dataset.")
        return

    # Calculate the mean of matching tracks' feature values to represent the song
    song_features = matching_tracks[numerical_features].mean().values.reshape(1, -1)

    # Calculate cosine similarity between the song and all other tracks
    song_similarity = cosine_similarity(song_features, data[numerical_features])

    # Get the indices of most similar tracks
    recommended_track_indices = song_similarity.argsort()[0][::-1][1:num_recommendations + 1]

    return data.iloc[recommended_track_indices][['artist', 'title']]


# Display recommendations
if st.sidebar.button("Recommend"):
    recommendations = recommend_tracks_by_name(song_name, num_recommendations)
    if recommendations is not None:
        st.header("🎶 Recommended Tracks")
        st.table(recommendations)

# Custom CSS for colorful and creative interface
st.markdown(
    """
    <style>
    body {
        background-color: #F7CAC9;
    }
    .st-bc {
        background-color: #FF6B6B;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-cv {
        background-color: #63D471;
        color: #FFFFFF;
        font-size: 24px;
        padding: 10px;
        border-radius: 0.5rem;
        text-align: center;
    }
    .st-bs {
        background-color: #FFD700;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-ek {
        background-color: #6495ED;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-co {
        color: #FF6B6B;
    }
    </style>
    """,
    unsafe_allow_html=True
)
