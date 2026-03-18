# Implementation-Of-Movie-Recommendation-System-Using-Machine-Learning-And-Deep-Learning
## 1. Introduction
Recommendation systems play a crucial role in modern platforms such as Netflix and Amazon by delivering personalized content to users. This project focuses on building and evaluating a movie recommendation system using both Machine Learning (ML) and Deep Learning (DL) approaches.
The goal is to compare traditional and modern techniques in terms of:
Prediction accuracy
Computational efficiency

## 2. Problem Statement
Traditional recommendation systems often struggle with:
Data sparsity
Cold-start problems
Scalability issues

While Deep Learning models can capture complex user-item interactions, they require high computational resources.
This project aims to answer:
Which model provides the best trade-off between accuracy and computational efficiency for movie recommendation systems?

## 3. Dataset
This project uses the MovieLens "ml-latest-small" dataset provided by GroupLens.
Dataset Details:
610 users
9,742 movies
100,836 ratings
3,683 tags

Files used:
ratings.csv – User ratings
movies.csv – Movie titles & genres
tags.csv – User-generated tags
links.csv – External IDs (IMDb, TMDb)

## 4. Process Workflow
The system follows the pipeline below:
Data Collection
Data Preprocessing
Merge datasets
Handle missing values

One-hot encode genres
Train-Test Split
80% training
20% testing

## 5. Methodology And Algorithms Used
Evaluation Metrics:
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
Training Time (Efficiency)

Algorithms Used
Machine Learning Models
KNNWithMeans
Singular Value Decomposition (SVD)
SVD++
Non-Negative Matrix Factorization (NMF)
BaselineOnly

Deep Learning Models
Neural Collaborative Filtering (NCF)
Wide & Deep Model
Transformer Model

## 6. Results
SVD++ achieved the best overall accuracy with the lowest RMSE (0.864) and MAE (0.662), though it required high training time (83.13s). The Transformer model showed similar accuracy (RMSE 0.870, MAE 0.665) but had the highest computational cost (92.70s). BaselineOnly was the fastest model (0.11s) while maintaining reasonable accuracy, making it highly efficient. SVD provided a good balance between performance and speed. In contrast, KNNWithMeans, NMF, NCF, and Wide & Deep showed higher error rates, with Wide & Deep performing the worst. Overall, traditional ML models outperformed DL models on this dataset, especially in terms of efficiency.

## 7. Explanation 
What I Did
Preprocessed MovieLens dataset using Pandas & NumPy
Implemented ML models using Surprise / Scikit-learn
Built DL models using:
TensorFlow (NCF, Wide & Deep)
PyTorch (Transformer)


Performed feature engineering (genre encoding)
Evaluated models using multiple metrics
Compared accuracy vs computational cost
Visualized results using:
Bar plots
Heatmaps
Line graphs

## 8. Key Insights
Best Accuracy: SVD++
Fastest Model: BaselineOnly
Best Trade-off: BaselineOnly & SVD
DL models did not outperform ML on this dataset


## 9. Challenges
Data sparsity in the user-item matrix
Limited computational resources
High training time for DL models
Small dataset (100K vs 25M limitation)


## 10. Future Work
Hybrid recommendation systems (ML + DL)
Context-aware recommendations (user behaviour, demographics)
Model optimization (pruning, quantization)
Cloud-based scalable deployment
Real-time recommendation updates






