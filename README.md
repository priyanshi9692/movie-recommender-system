# movie-recommender-system

## Deployable model as a service

* Build Docker Image

`docker build --tag recommender-system .`

* Run the application

`docker run -p 8501:8501 recommender-system:latest`

* Go to http://localhost:8501

* Enter User Id for any user to see recommendation

    ![streamlit app](streamlitapp-ui.jpeg)

## GitRepo package contains

* Datasets: `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv` (provided in the assessment itself) `tmdb_metadata.csv` (created using script mentioned in the `data-analysis-report.ipynb` file)
* Dockerfile
* requirements.txt
* streamlit application code (`streamlit_recommender.py`)
* flask API server code (`recommender_movie_rest_server.py`)
* Data analysis report (ipynb file as well as pdf)
* Business evaluation report (ipynb file as well as pdf)
* Community recommendation engine experiment report (ipynb file as well as pdf)
