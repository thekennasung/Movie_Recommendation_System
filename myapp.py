import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Function to display movie posters
def display_image(id):
    return f'https://liangfgithub.github.io/MovieImages/{id}.jpg'

# myIBCF function
def myIBCF(new_user_ratings, smat, movie_ids, popularity_scores):
    predictions = pd.Series(np.nan, index=movie_ids)

    to_be_rated, rated = [], []
    
    for k in new_user_ratings.keys():
        if np.isnan(new_user_ratings[k]):
            to_be_rated.append(k)
        else:
            rated.append(k)

    for movie_id in to_be_rated:
        #print(movie_id, new_user_ratings)
        similar_movies = smat.loc[movie_id].dropna() if movie_id in smat.index else pd.Series(dtype=float)
        #rated_movies = new_user_ratings[~new_user_ratings.isna()].index
        common_movies = similar_movies.index.intersection(rated)

        new_user_ratings_score = []

        for cv in common_movies:
            new_user_ratings_score.append(new_user_ratings[cv])

        new_user_ratings_score = np.array(new_user_ratings_score)

        if len(common_movies) > 0:
            numerator = (similar_movies[common_movies] * new_user_ratings_score).sum()
            denominator = similar_movies[common_movies].sum()

            if denominator > 0:
                predictions[movie_id] = numerator / denominator

    recommendations = predictions.sort_values(ascending=False).dropna().index.tolist()
    
    if len(recommendations) < 10:
        unrated_popular_movies = popularity_scores.loc[
            ~popularity_scores['MovieID'].isin(new_user_ratings[~new_user_ratings.isna()].index)
        ]['MovieID'].tolist()
        recommendations.extend(unrated_popular_movies[:10 - len(recommendations)])
    
    top_10 = predictions.loc[recommendations[:10]].reset_index(name='PredictedRating')
    top_10 = top_10.rename(columns={'Unnamed: 0': 'MovieID'}).fillna(0)

    return top_10

# Load data
url_ratings = "https://raw.githubusercontent.com/liangfgithub/liangfgithub.github.io/master/MovieData/ratings.dat"
url_movies = "https://raw.githubusercontent.com/liangfgithub/liangfgithub.github.io/master/MovieData/movies.dat"
ratings = pd.read_csv(url_ratings, sep = '::', engine = 'python', header = None)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
movies = pd.read_csv(url_movies, sep = '::', engine = 'python',
                     encoding = "ISO-8859-1", header = None)
movies.columns = ['MovieID', 'Title', 'Genres']

# Calculate the popularity score
# Group Ratings by MovieID
avg_ratings = ratings.groupby('MovieID').mean()['Rating']
count_ratings = ratings.groupby('MovieID').count()['Rating']
movies = movies.join(avg_ratings, on='MovieID')
movies = movies.join(count_ratings, on='MovieID', lsuffix='_avg', rsuffix='_count')
# Normalize
movies['rating_avg_norm'] = (movies['Rating_avg'] - movies['Rating_avg'].min()) / (movies['Rating_avg'].max() - movies['Rating_avg'].min())
movies['rating_count_norm'] = (movies['Rating_count'] - movies['Rating_count'].min()) / (movies['Rating_count'].max() - movies['Rating_count'].min())
# Calculate the popularity score
movies['score'] = movies['rating_avg_norm'] * 0.5 + movies['rating_count_norm'] * 0.5
popularity_scores = movies[['MovieID', 'score']].sort_values(by = 'score', ascending = False)

# Load top30 smat
smat = pd.read_csv("top30_smat.csv")
smat.set_index('Unnamed: 0', inplace=True)
movie_ids = smat.index

movies['Poster'] = movies['MovieID'].apply(display_image)

# Streamlit UI
st.title("STAT542 -- Movie Recommendation System")

# Step 1: Display sample movies and collect ratings
st.header("Rate as many movies as possible")
sample_movies = movies.sample(50, random_state = 50)
new_user_ratings = pd.Series(index=movie_ids, data=np.nan)

# Display movies in a grid 
cols_per_row = 5
rows = len(sample_movies) // cols_per_row + 1

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        index = i * cols_per_row + j
        if index < len(sample_movies):
            movie = sample_movies.iloc[index]
            with cols[j]:
                st.image(movie['Poster'], width=120)
                st.markdown(f"**{movie['Title']}**")
                rating = st.slider("Rate this movie:", 0, 5, 0, 1, key=movie['MovieID'])
                new_user_ratings['m'+str(movie['MovieID'])] = rating

# Step 2: Generate and display recommendations
if st.button("Submit Ratings"):
    if new_user_ratings.sum() == 0:
        st.warning("Please rate at least one movie to get recommendations.")
    else:
        new_user_ratings_formatted = {}
        for movie_id in new_user_ratings.index:
            formatted_id = f"{movie_id}"
            new_user_ratings_formatted[formatted_id] = new_user_ratings[movie_id]

        recommendations = myIBCF(new_user_ratings_formatted, smat, movie_ids, popularity_scores)
        print(recommendations)
        # Display recommendations with poster and title
        st.header("Top Movie Recommendations for You")
        cols_per_row = 5
        rows = len(recommendations) // cols_per_row + 1

        movie_id = [ int(s.replace('m', '')) for s in recommendations['MovieID']]
        print(movie_id)
        movie_poster, movie_title = [], []
        for id in movie_id:
            m = movies[movies["MovieID"]== id]
            movie_poster += m['Poster'].to_list()
            movie_title += m['Title'].to_list()

        #print(movie_poster, movie_title)
        ind = 0
        
        for i in range(rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                index = i * cols_per_row + j
                if index < len(recommendations):
                    with cols[j]:
                        st.image(movie_poster[ind], width=120)
                        st.markdown(f"**{movie_title[ind]}**")
                        ind += 1
