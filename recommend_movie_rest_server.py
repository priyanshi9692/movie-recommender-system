"""
Movie Recommender System Deployment Script

This script deploys a movie recommender system using a pre-trained model.
It exposes an API endpoint '/predict' to provide movie recommendations for a given user.

Author: Priyanshi Jajoo

Dependencies:
- Flask
- pandas
- joblib

Usage:
1. Ensure that the necessary libraries are installed: `pip install flask pandas joblib`
2. Save the trained recommendation model as 'recommendation_model.joblib' in the same directory.
3. Run the script using: `python script_name.py`
4. Access recommendations for a specific user by making a GET request to '/predict' with the 'user_id' parameter.

Example API Request:
GET http://localhost:8080/predict?user_id=123

Response:
[("Movie Title 1", estimated_rating_1), ("Movie Title 2", estimated_rating_2), ...]

Note:
- This script assumes the presence of 'ratings.csv' and 'movies.csv' files in the working directory.
- Adjust the 'port' and 'debug' parameters in `app.run()` for production use.

"""
from joblib import load
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load('recommendation_model.joblib')

@app.route('/predict')
def get_user_recommendations():
    """
    Endpoint to get movie recommendations for a specified user.

    Returns:
    - JSON response containing a list of recommended movies with estimated ratings.
    """
    user_id= (int)(request.args.get('user_id'))
    recommendations = get_movie_recommendations(user_id, model, n=5)
    print(recommendations)
    return recommendations


def get_movie_recommendations(user_id, model, n):
    """
    Generates movie recommendations for a given user.

    Parameters:
    - user_id (int): ID of the user for whom recommendations are requested.
    - model: Pre-trained recommendation model.
    - n (int): Number of movie recommendations to generate.

    Returns:
    - List of tuples containing movie titles and their estimated ratings.
    """

    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    
    all_movie_ids = ratings['movieId'].unique()
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    user_unrated_movies = []
    for movie_id in all_movie_ids:
        if movie_id not in user_rated_movies:
            user_unrated_movies.append(movie_id)

    print(len(user_unrated_movies))
    # Predict ratings for unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in user_unrated_movies]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top N recommendations
    top_recommendations = predictions[:n]
    top_movies_info = [(movies[movies['movieId'] == prediction.iid]['title'].values[0], prediction.est) for prediction in top_recommendations]
    return top_movies_info


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080) 
