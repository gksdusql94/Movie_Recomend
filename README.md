# Movie Recommendation System Using Spark ALS

## Overview

This project implements a movie recommendation system using the Alternating Least Squares (ALS) algorithm in PySpark. The system is based on the MovieLens dataset, which includes 100,000 ratings provided by 1,000 users on 1,700 movies. The ALS algorithm is a matrix factorization technique widely used in collaborative filtering for recommendation systems.

## Project Structure

The project consists of the following key steps:

1. **Data Loading and Preprocessing**  
   - The MovieLens dataset is split into a training set (80%) and a test set (20%).
   - Movie data is loaded and mapped to corresponding titles using the item file.

2. **Model Development**  
   - The ALS algorithm is used to build a recommendation model on the training data.
   - Regularization (`regParam`) and other hyperparameters are fine-tuned for optimal performance.

3. **Evaluation**  
   - The model's performance is evaluated using Root Mean Squared Error (RMSE) on the test set.

4. **Recommendation Generation**  
   - Top-5 movie recommendations are generated for all users based on their preferences and viewing history.

## Dataset

- **Source**: [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/)
- **Content**: 
  - 100,000 ratings from 1,000 users on 1,700 movies.
  - Each rating is a value from 1 to 5, representing the user's preference.

## Results

- The system achieved a **Root Mean Squared Error (RMSE)** of **0.938** on the test dataset.
- Successfully generated **top-5 personalized movie recommendations** for all users.

## Installation and Setup

To run the project on your local machine or a cloud environment like Google Colab, follow these steps:

1. **Install PySpark**  
   You can install PySpark and necessary dependencies using:
   ```bash
   !pip install pyspark

2. **Set Up the Environment**

Make sure you have a Java environment (Java 8) for running PySpark. In Colab, you can install it using:

```bash
!apt install openjdk-8-jdk-headless -qq
3. **Download the Dataset**
The MovieLens dataset is available online. You can either download it directly or use the provided links in the notebook to fetch the training, test, and movie items data.

4. **Run the code**
After setting up the environment and loading the dataset, run the provided code to train the ALS model and generate recommendations.


## How to Use
Training the Model
Run the notebook to load and preprocess the dataset, then fit the ALS model on the training data.

Evaluate the Model
After training, the model is evaluated using RMSE on the test dataset.

Generate Recommendations
The model will generate the top-5 recommendations for each user based on their past ratings.
