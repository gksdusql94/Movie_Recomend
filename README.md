# 🎥 Movie Recommendation System Using Spark ALS

## 📖 Overview
This project implements a movie recommendation system using the **Alternating Least Squares (ALS)** algorithm in PySpark. The system is based on the **MovieLens dataset**, which includes 100,000 ratings provided by 1,000 users on 1,700 movies. The ALS algorithm is a matrix factorization technique widely used in collaborative filtering for recommendation systems.

### Key Highlights:
- Achieved an **RMSE (Root-Mean-Square Error)** of **0.938** on the test dataset using Spark's ALS model.
- Trained on 80,000 samples and tested on 20,000 samples from the dataset.
- Generated top 5 movie recommendations for all users, optimizing the personalized experience.

---

## 📂 Project Structure

The project follows these key steps:

1. **Data Loading and Preprocessing**  
   - The MovieLens dataset is split into a **training set (80%)** and a **test set (20%)**.
   - Movie data is mapped to corresponding movie titles using the `items` file.

2. **Model Development**  
   - The **ALS algorithm** is applied to build the recommendation model.
   - Regularization (`regParam`) and other hyperparameters are fine-tuned for optimal performance.

3. **Evaluation**  
   - The model's performance is evaluated using **Root Mean Squared Error (RMSE)** on the test set.

4. **Recommendation Generation**  
   - Top-5 personalized movie recommendations are generated for each user based on their past ratings and preferences.

---

## 📊 Dataset

- **Source**: [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/)
- **Content**:  
  - 100,000 ratings from 1,000 users on 1,700 movies.
  - Each rating is on a scale of **1 to 5**, representing the user's preference for the movie.

---

## 📈 Results and Visualizations

### 1. **RMSE Evaluation**:
The **Root Mean Squared Error (RMSE)** is used to evaluate the model's accuracy on the test dataset. A lower RMSE indicates better performance. The final RMSE achieved is **0.938**.

#### RMSE Plot Example:
You can visualize the **RMSE** for different iterations of the model using `matplotlib`:

```python
import matplotlib.pyplot as plt

iterations = [5, 10, 15, 20]
rmse_values = [1.05, 0.98, 0.95, 0.938]

plt.figure(figsize=(8, 6))
plt.plot(iterations, rmse_values, marker='o')
plt.title('RMSE vs Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()
```

### 2. User Recommendations Visualization:
Once the model is trained, we can visualize the top-5 movie recommendations for a specific user. Here's an example using matplotlib to show the recommended movies:

```python
Copy code
import pandas as pd

# Example user recommendation data
user_recommendations = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Rating': [4.8, 4.6, 4.5, 4.4, 4.2]
}

df = pd.DataFrame(user_recommendations)
df.plot(kind='barh', x='Movie', y='Rating', color='skyblue', legend=False)
plt.title('Top 5 Movie Recommendations for User')
plt.xlabel('Predicted Rating')
plt.ylabel('Movie')
plt.show()
```

## ⚙️ Installation and Setup
To run this project on your local machine or a cloud environment like Google Colab, follow these steps:

1. **Install PySpark**
You can install PySpark and necessary dependencies using:

```bash
!pip install pyspark
```

2. **Set Up the Environmentt**  
Make sure you have Java 8 installed for running PySpark. In Colab, you can install it with:
```bash
!apt install openjdk-8-jdk-headless -qq
```

3. **Download the Dataset**  
Download the MovieLens dataset using the provided Google Drive links or directly from the MovieLens website @https://grouplens.org/datasets/movielens/

4. **Run the Code**  
After setting up the environment and loading the dataset, run the provided code to train the ALS model and generate recommendations.


## 💻 How to Use
1. **Training the Model**  
   Run the notebook to load and preprocess the dataset, then fit the ALS model on the training data.

2. **Evaluating the Model**  
   After training, the model is evaluated using RMSE on the test dataset. You can visualize the RMSE results as shown above.

3. **Generating Recommendations**  
   The model will generate the top-5 movie recommendations for each user based on their past ratings and preferences.

## 🎯 Conclusion  
The system achieved an RMSE of 0.938 on the test dataset.  
Successfully generated top-5 personalized movie recommendations for all users.  
The ALS algorithm proves to be an effective method for collaborative filtering in recommendation systems.

```result
+-------+--------------------------------------------------------------------------------------------+
|user_id|recommendations                                                                             |
+-------+--------------------------------------------------------------------------------------------+
|1      |[{1449, 4.96}, {127, 4.92}, {1368, 4.92}, {169, 4.88}, {408, 4.88}]                         |
|2      |[{1643, 5.17}, {1449, 5.07}, {318, 4.89}, {863, 4.89}, {427, 4.84}]                         |
|3      |[{1643, 5.57}, {641, 5.14}, {199, 4.84}, {320, 4.75}, {1524, 4.59}]                         |
|4      |[{745, 5.87}, {1005, 5.82}, {1159, 5.78}, {372, 5.76}, {1192, 5.72}]                        |
|5      |[{50, 4.67}, {1142, 4.61}, {115, 4.60}, {390, 4.58}, {169, 4.55}]                           |
+-------+--------------------------------------------------------------------------------------------+

```
