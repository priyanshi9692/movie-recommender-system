"""
Movie Recommender System Deployment using Streamlit application
Author: Priyanshi Jajoo
"""
import pandas as pd
import streamlit as st
from joblib import load

# Load the trained model
model = load('recommendation_model.joblib')
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

def get_movie_recommendations(user_id, model, n=5):
    # Get a list of all movie IDs
    all_movie_ids = ratings['movieId'].unique()

    # Get movie IDs not rated by the user
    user_ratings = ratings[ratings['userId'] == user_id]
    user_rated_movies = user_ratings['movieId'].values
    user_unrated_movies = []
    for movie_id in all_movie_ids:
        if movie_id not in user_rated_movies:
            user_unrated_movies.append(movie_id)

    # Predict ratings for unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in user_unrated_movies]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top N recommendations
    top_recommendations = predictions[:n]
    top_movies_info = [(movies[movies['movieId'] == prediction.iid]['title'].values[0], prediction.est) for prediction in top_recommendations]
    
    return top_movies_info

"""
    Display movie recommendations for a user entered through a Streamlit app.

    A user-entered ID, retrieves movie recommendations using a pre-trained model,
    and displays the top movie recommendations along with their ratings in a formatted table.

    Usage:
    1. Enter the user ID in the Streamlit app.
    2. If a valid user ID is provided, the function fetches movie recommendations and displays them.
    3. The table is styled for better readability.
"""
user_id = st.text_input(label="User Id")

if user_id:
    recommendations = get_movie_recommendations(int(user_id), model, n=5)
    columns = ["Movies", "Ratings"]
    df = pd.DataFrame(recommendations, columns=columns)
    st.write("Top Movie Recommendations:")
    st.table(df.style.set_table_styles([{
        'selector': 'thead th',
        'props': [('text-align', 'center')]
    }]).set_properties(**{'text-align': 'center'}))
else:
    st.write("Please enter your user id for your next movie recommendation :) ")
