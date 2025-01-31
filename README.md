# Movie Recommendation System

## Overview
An intelligent movie recommendation system built using Python, Pandas, and Streamlit. This project applies collaborative filtering techniques to provide personalized movie suggestions based on user ratings.

## Features
- **Interactive Web Interface**: Built with Streamlit to provide an intuitive and user-friendly experience.
- **Personalized Recommendations**: Utilizes item-based collaborative filtering to suggest movies based on user ratings.
- **Movie Poster Integration**: Fetches and displays movie posters using an external API.
- **Popularity-Based Suggestions**: Ensures recommendations even with minimal user input.

## Data Source
This project utilizes the **MovieLens 1M Dataset**, which contains 1 million ratings from users on various movies. The similarity matrix is computed using the user rating matrix.

## Usage
1. Open the web application.
2. Rate as many movies as possible.
3. Click **Submit Ratings** to get recommendations.
4. View your personalized movie recommendations along with their posters.

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn)
- Streamlit (For the web-based interface)
- MovieLens 1M Dataset (For training recommendations)
- External API (For fetching movie posters dynamically)
