# 🎥 Movie Recommendation System Using Spark ALS

## 📖 Overview
This project implements a movie recommendation system using the Alternating Least Squares (ALS) algorithm in PySpark. The system is based on the MovieLens dataset, which includes 100,000 ratings provided by 1,000 users on 1,700 movies. The ALS algorithm is a matrix factorization technique widely used in collaborative filtering for recommendation systems.

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

## ⚙️ Python Code Implementation
### 1. Installing Required Packages and Setting Up Environment:
```python
!pip install pyspark
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

### 2. Data Loading:
The dataset is loaded into Spark DataFrames for training and testing. The schema for the ratings and items is defined as follows:

```python
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

schema_ratings = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("item_id", IntegerType(), False),
    StructField("rating", IntegerType(), False),
    StructField("timestamp", IntegerType(), False)
])

schema_items = StructType([
    StructField("item_id", IntegerType(), False),
    StructField("movie", StringType(), False)
])

# Load data into Spark DataFrames
training = spark.read.option("sep", "\t").csv("MovieLens.training", header=False, schema=schema_ratings)
test = spark.read.option("sep", "\t").csv("MovieLens.test", header=False, schema=schema_ratings)
items = spark.read.option("sep", "|").csv("MovieLens.item", header=False, schema=schema_items)
```

### 3. Model Training Using ALS:
The Alternating Least Squares (ALS) model is applied to the training data to generate movie recommendations.
```python
from pyspark.ml.recommendation import ALS

# Set ALS parameters
als = ALS(maxIter=20, regParam=0.09, userCol="user_id", itemCol="item_id", ratingCol="rating", coldStartStrategy="drop")

# Train the ALS model
model = als.fit(training)
])

# Load data into Spark DataFrames
training = spark.read.option("sep", "\t").csv("MovieLens.training", header=False, schema=schema_ratings)
test = spark.read.option("sep", "\t").csv("MovieLens.test", header=False, schema=schema_ratings)
items = spark.read.option("sep", "|").csv("MovieLens.item", header=False, schema=schema_items)
```
### 4. Model Evaluation with RMSE:
To evaluate the performance of the ALS model, Root Mean Squared Error (RMSE) is calculated on the test set.
```python
from pyspark.ml.evaluation import RegressionEvaluator

# Make predictions on the test data
predictions = model.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```
### 5. Generating Top-5 Movie Recommendations for All Users:
Finally, the model is used to generate the top-5 movie recommendations for each user in the dataset.
```python
top_K = 5
userRecs = model.recommendForAllUsers(top_K)

# Display top-5 recommendations for 5 users
userRecs.show(5, False)
```
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
