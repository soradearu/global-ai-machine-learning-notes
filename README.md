# GLOBAL AI HUB MACHINE LEARNING BOOTCAMP NOTES
![Generic badge](https://img.shields.io/badge/machine-learning-green.svg) ![Generic badge](https://img.shields.io/badge/AI-blue.svg)  [![Generic badge](https://img.shields.io/badge/globalaihub-blue.svg)](https://globalaihub.com/)
<hr/>
A repository that contains my notes from Global AI Hub Machine Learning Bootcamp mashed with my own research. I explained things the way I understand and tried to keep it simple. You can make pull requests and add more notes if you'd like. i will keep adding more data as I expand my knowledge on the subject.
<br>



## Table of Contents

- [What is Machine Learning?](#what-is-machine-learning)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Regression](#regression)
- [Decision Trees](#decision-trees)
- [Classification](#classification)
- [Books](#books)



## What is Machine Learning?
 Machine learning is like teaching computers to learn from examples, just like how we learn from our day to day experiences. We show the computer lots of examples and tell it what's right or wrong. It uses these examples to get better at doing things on its own, like making predictions or recognizing patterns. It's like training a computer to be smart by showing it hundreds of pictures, numbers, or words.


## Supervised Learning

Supervised learning is a subcategory of machine learning. You show the program lots of examples, pictures, numbers, but in this case, we use data with labels. The program learns from these examples. When the program learns enough, we can give it new examples it hasn't seen before, and it will try to guess what the right label is.

- Key points
  - **Clustering** : Clustering in machine learning is the process of grouping similar data points together in order to find patterns and structures within a dataset.
  - **Training Data** : Training data is the information used to teach a computer model how to make predictions or decisions.
  - **Prediction** : Prediction in machine learning is like guessing what might happen in the future based on patterns and information from the past.
  - **Classification** : It is teaching computers to recognize and categorize things based on their features.
  - **Regression** : It is a way to figure out the relationship between variables and make predictions based on that relationship.
  - **Decision Boundary** : A decision boundary in machine learning is like an imaginary line that separates different categories or classes of data.
  - **Overfitting**: Overfitting in machine learning is when a model learns the training data too well, capturing noise and making it perform poorly on new, unseen data.
  - **Underfitting** : Underfitting in machine learning is when a model is too simple to capture the patterns in the data, resulting in poor performance.
  - **Bias-Variance Trade-off** : The bias-variance trade-off in machine learning is about finding the right balance between making simple but possibly inaccurate predictions (bias) and making complex but possibly overfit predictions (variance). 
  - **Ensemble Learning** : Ensemble learning in machine learning is when multiple models work together to make a more accurate prediction than any single model could.
  - **Test Data** : Test data in machine learning is a separate set of examples that the model hasn't seen before, used to check how well the model can make accurate predictions on new, unseen information.
  - **Feature**: A feature in machine learning is a characteristic or property of the data that helps the model understand and make predictions.
  - **Validation Data**: Validation data in machine learning is a separate set of data used to check how well a model performs on new, unseen examples.
  - **Accuracy** : Accuracy in machine learning measures how often a model's predictions are correct, showing how well it's doing its job.
  - **Precision** : Precision in machine learning measures how accurate the positive predictions are among all the positive predictions made by a model.
  - **Recall** : Recall in machine learning is the measure of how well a model finds all the relevant instances among all the actual instances.
  - **F1 Score** : The F1 score in machine learning measures the balance between precision and recall, helping us understand how well a model finds relevant results while avoiding false positives and false negatives.
  - **Random Forest** : Random Forest in machine learning is like asking a bunch of experts (decision trees) for advice and combining their opinions to make more accurate predictions.
  - **Support Vector Machine (SVM)**: Support Vector Machine (SVM) in machine learning is like finding the best line that separates different groups of data points.
  - **Cross-Validation** : Cross-validation in machine learning is a technique to test how well a model works by splitting the data into different parts and checking its performance on each part.

## Unsupervised Learning

Unsupervised learning is a bit of an adventure. Imagine you are in library and books doesn't have labels on them. So you will need to sort them out. So you look at each and every book sort them by category, names, published dates, writers. You try to find similarities and try to find the best way the organise all the books so next time you will find whatever you are looking for faster.

This is exactly what you do in unsupervised learning. There are lots of data without any labels. Program looks at this data and tries to find patterns and similarities all by itself. It groups similar things together. The program keeps sorting and grouping until it thinks it found the best way to organize the data.

- Key Points
  - **Clustering** : Clustering in machine learning is the process of grouping similar data points together in order to find patterns and structures within a dataset.
  - **PCA (Principal Component Analysis)**
  - **t-SNE**: t-SNE (t-Distributed Stochastic Neighbor Embedding) is a machine learning technique that helps visualize and cluster high-dimensional data points in a lower-dimensional space while preserving their similarities.
  - **Centroid**: A centroid in machine learning is like the average position of a group of data points, helping us understand the center of a cluster.
  - **Hierarchical Clustering**: Hierarchical Clustering is a method to group data points based on their similarity, creating a tree-like structure of clusters known as dendogram.
  - **Silhouette Score** : It measures how close each data point is to its own cluster compared to other clusters.
  - **Outlier** : An outlier is a data point that's very different from the others and might affect the accuracy of predictions.
  - **Gaussian Mixture Model** : A Gaussian Mixture Model is mixing different bell-shaped curves to describe complex patterns in data that might be made up of multiple underlying groups.
  - **Anomaly Detection** : Anomaly detection is finding something unusual or unexpected in a group of things.
  - **Dimensionality Reduction** : Dimensionality Reduction is simplifying a complicated puzzle by keeping only the important pieces, so it's easier to see patterns and understand the big picture.
  - **K-means** : K-means is a way to group similar data points together into clusters by finding common patterns.
  - **Mini-Batch K-means** : It is a faster way to group similar data points into clusters by processing small batches of data at a time instead of the entire dataset.
  - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** : DBSCAN, is finding groups of points that are close to each other in a dataset, while also identifying points that don't belong to any group and might be outliers.
  - **Agglomerative Clustering** :Agglomerative Clustering is joining the most similar data points together step by step to create groups. 
  - **Divisive Clustering** : Divisive Clustering, is starting with all the data in one big group and then splitting it into smaller groups step by step to find similarities within the data.
  - **Affinity Propagation Algorithm** : APA, is like a group chat where data points decide which point is the best representative for their group based on similarity and communication.
  - **Elbow Method** :  It helps us find the best number of groups in clustering by looking for the "bend" point where adding more groups doesn't significantly improve the model.
  - **Silhouette Score** : It measures how well-separated clusters are in data, helping us know if the clusters are distinct and well-defined.

## Regression
Imagine you're running a pizza delivery business, and you want to figure out how long it will take for a pizza to be delivered based on the distance between your shop and the customer's location. Regression in machine learning is like using past delivery data to create a mathematical rule that helps you estimate delivery time. You look at the data for past deliveries, where you know the distance and delivery time, and you find a pattern. This pattern helps you predict future delivery times based on the distance, even for new customers. 
  - **Prediction Tool** :
  - **Linear Relationship** :
  - **Best-Fit Line** :
  - **Coefficients** :
  - **Ordinary Least Squares (OLS)** :
  - **Overfitting and Underfitting** :
  - **Residuals** :
  - **R-squared** :
  - **Types of Regression** :
  - **Assumptions** :
  - **Multicollinearity** :
  - **Interpretability** :
  - **Evaluation Metrics** :
  - **Real-world Applications** :
  - **Improvement Techniques** :
  - **Limitations** :


## Decision trees
Decision tree is a tool that helps computers make choices. Instead of asking you questions, it uses facts and features to figure things out. The computer starts with a question, like "Is it red?" If the answer is yes, it might guess it's an apple. If the answer is no, it could guess it should be something else.
By asking a series of questions and getting answers, the computer can reach a decision, just like you would figure out what to do using a flowchart.

- Key Points
  - Root
  - Node
  - Leaf
  - Interval Nodes
  - Depth
  - Overfitting
  - Underfitting
  - Gini Index
  - Pure
  - Impure
  - Entropy
  - Max Depth
  - Bootstrapping
  - Bootstrap Aggregation or Bagging
  - Random Forest
  - Boosting


## Classification

Imagine you're a character in a zombie survival game where you encounter different kinds of zombies. Each zombie belongs to a specific class like "Normal", "Mutated","Special" And you're equipped with a special device that can analyze a zombie's appearance and behavior and determine its class so you will have more chance to survive if you encounter one with more information.

Classification in machine learning is like training this device we have to recognize and classify these zombies automatically. 
  1. You go out there and gather more data about these zombies like their sizes, how they look,how big they are, do they sprint or walk. 
  2. After that you label them as "Normal", "Special", "Mutated" and such. 
  3. Using the collected data and labels, you teach (train) a machine learning model how to associate features with zombies' classes.
  4. Once the model is trained, you can input the features of a new zombie it hasn't seen before. The model analyzes these features and predicts the class the zombie belongs to based on its learned patterns.


- Key points
  - Classification
  - Binary Classification
  - Logistic Regression
  - Sigmoid curve
  - Activation Functions
  - Thresholding
  - Confusion Matrix
  - Accuracy matrice
  - Precision
  - Recall
  - ROC curve
  - AUC (Area Under the Curve)
  - F! Score
  - Harmonic Mean
  - Random Forest
  - KNN (K-Nearest Neighbours)
  - Support Vector Machine
  - Hard margin
  - Soft Margin
    
  



## Books
* [Hand-On-Machine-Learning-With-Scikit-Learn-Keras-And-Tensorflow-By-Aurelion-Geron]()
* [Machine-Learning-With-Python-Cookbook-Chris-Albon]()




## Other Resources

