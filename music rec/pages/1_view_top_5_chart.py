import streamlit as st
import pandas as pd

df= pd.read_csv("Spotify.csv")
df= df.iloc[0:999]

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
    df["top genre"].replace(key, value, inplace = True)

df=df.drop_duplicates(subset="title")


st.header("Top 5's")

st.write("Top 5 Artist")
top_5_art= df[['artist', 'title']].groupby("artist").count().sort_values(by="title", ascending=False)[:5]
st.bar_chart(top_5_art)

st.write("Top 5 Genere")
top_5_genre= df[['top genre', 'title']].groupby("top genre").count().sort_values(by="title", ascending=False)[:5]
st.bar_chart(top_5_genre)

st.write("Top 5 Dancability songs")
top_5_dnce= df[['dnce', 'title']].groupby("title").count().sort_values(by="dnce", ascending=False)[:5]
st.bar_chart(top_5_dnce)

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


st.write("Word Cloud of Genre")
a = df['top genre'].value_counts() 
text1 = a

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate_from_frequencies(text1)

# Display the generated image:
plt.figure( figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()


st.write("Word Cloud of Artist")
b = df['artist'].value_counts() 
text2 = b

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate_from_frequencies(text2)

# Display the generated image:
plt.figure( figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
st.pyplot()

