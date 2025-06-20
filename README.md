# Machine Learning Algorithms Implementation

This repository contains various machine learning algorithm implementations, from basic decision trees to more complex models like XGBoost. Each implementation includes detailed code with explanatory comments to help understand the underlying concepts.

## Table of Contents
- [Decision Trees and XGBoost](#decision-trees-and-xgboost)
  - [Decision Tree Implementation](#decision-tree-implementation)
  - [Ads Click-Through Rate Prediction](#ads-click-through-rate-prediction)
- [Support Vector Machines](#support-vector-machines)
  - [Face Recognition](#face-recognition)
- [Naive Bayes](#naive-bayes)
  - [Spam Detection](#spam-detection)
  - [Movie Recommendation](#movie-recommendation)
- [Logistic Regression](#logistic-regression)
  - [Ads Click-Through Rate Prediction](#logistic-regression-ads-click-through)

## Decision Trees and XGBoost

### Decision Tree Implementation
**File:** `/XGBoost/decisionTreeImplementation.py`

This file contains a comprehensive implementation of decision trees from scratch, showcasing the core principles of how decision trees work.

**Key Components:**

1. **Impurity Metrics Implementation**:
   - Gini impurity and entropy calculations with visualizations
   - Weighted impurity calculation for evaluating splits

2. **Decision Tree Node Split Functions**:
   - `split_node()`: Splits data based on feature and value
   - `get_best_split()`: Finds optimal feature and threshold for splitting
   - `split()`: Recursively builds the tree by splitting nodes

3. **Training Function**:
   - `train_tree()`: Main function to build the tree from training data
   - Configurable parameters for max depth and minimum samples for splits

4. **Tree Visualization**:
   - Custom visualization function to display the decision boundaries
   - Comparison with scikit-learn's implementation

The implementation handles both numerical and categorical features, demonstrating how decision trees can work with different data types. The example showcases binary classification problems with both categorical features (tech, fashion, sports) and numerical features.

### Ads Click-Through Rate Prediction
**File:** `/XGBoost/adsClickThroughRate.py`

This file demonstrates using XGBoost for predicting click-through rates for online advertisements.

**Key Features:**

1. **Data Loading and Preprocessing**:
   - Loads advertisement click-through data
   - Removes non-essential features like IDs and timestamps
   - One-hot encodes categorical features

2. **XGBoost Model**:
   - Creates and trains an XGBoost classifier with optimized parameters
   - Uses parameters: learning_rate=0.1, max_depth=10, n_estimators=1000

3. **Evaluation**:
   - Evaluates model performance using ROC AUC score
   - Provides probability predictions for click-through events

This implementation demonstrates how gradient boosting can be applied to real-world advertising data to predict user behavior.

## Support Vector Machines

### Face Recognition
**File:** `/SVM/faceRecognition.py`

This implementation uses Support Vector Machines (SVM) for face recognition tasks with the Labeled Faces in the Wild (LFW) dataset.

**Key Components:**

1. **Data Loading and Exploration**:
   - Loads face image data with a minimum of 80 faces per person
   - Visualizes example faces from the dataset
   - Prepares training and test splits

2. **SVM Model Training**:
   - Implements grid search for hyperparameter optimization
   - Tests different kernels (rbf, linear) and parameters (C, gamma)
   - Uses class_weight='balanced' to handle potential class imbalance

3. **Performance Evaluation**:
   - Reports accuracy and detailed classification metrics
   - Provides a classification report with precision, recall, and F1-score

4. **Dimensionality Reduction Pipeline**:
   - Implements PCA for feature extraction before SVM
   - Creates a pipeline combining PCA and SVM
   - Demonstrates substantial performance improvement with dimensionality reduction

The implementation shows how combining PCA with SVM significantly improves face recognition performance by reducing noise and extracting the most informative features from face images.

## Naive Bayes

### Spam Detection
**File:** `/NaiveBayes/SpamDetection.ipynb`

This Jupyter notebook implements a Naive Bayes classifier for email spam detection.

**Key Components:**

1. **Data Processing**:
   - Loads emails from spam and ham (non-spam) folders
   - Text preprocessing: lemmatization, removal of stop words and names
   - Feature extraction using CountVectorizer

2. **Naive Bayes Implementation**:
   - Calculates prior probabilities of classes
   - Computes likelihoods of features given classes
   - Implements posterior probability calculation using Bayes' theorem
   - Handles log probabilities to avoid numerical underflow

3. **Model Evaluation**:
   - Tests model on held-out data
   - Creates confusion matrix visualization
   - Reports precision, recall, F1-score, and accuracy
   - Achieves approximately 93% accuracy on test data

4. **Additional Analysis**:
   - Visualizes feature frequency distributions
   - Shows most common terms in spam vs non-spam emails

The implementation demonstrates how Naive Bayes is particularly effective for text classification tasks like spam detection due to its ability to handle high-dimensional data efficiently.

### Movie Recommendation
**File:** `/NaiveBayes/movieRecc.py`

This file implements a Naive Bayes approach to movie recommendations, predicting whether a user will like a movie based on their ratings of other movies.

**Key Components:**

1. **Data Loading and Preprocessing**:
   - Loads MovieLens 1M dataset with user ratings
   - Creates a user-movie matrix of ratings
   - Converts ratings to binary preferences (like/dislike)

2. **Naive Bayes Classification**:
   - Uses MultinomialNB from scikit-learn
   - Predicts whether users will like a specific movie based on their other ratings
   - Explores different smoothing parameters (alpha) and prior settings

3. **Model Evaluation**:
   - Implements K-fold cross-validation for robust evaluation
   - Calculates ROC curves and AUC scores
   - Reports precision, recall, and F1-score metrics

4. **Hyperparameter Tuning**:
   - Tests different smoothing factors and prior probability settings
   - Finds optimal parameters via cross-validation
   - Demonstrates how parameter choices affect model performance

This implementation shows how Naive Bayes can be applied to collaborative filtering for recommendation systems, providing personalized predictions based on user rating patterns.

## Logistic Regression

### Logistic Regression Ads Click-Through
**File:** `/LogisticRegression/AdsClickThrough.py`

This file implements logistic regression from scratch for predicting online advertisement click-through rates.

**Key Components:**

1. **Core Logistic Regression Functions**:
   - `sigmoid()`: Implements the sigmoid activation function
   - `compute_prediction()`: Calculates probability predictions
   - `compute_cost()`: Implements cross-entropy loss function

2. **Optimization Algorithms**:
   - `update_weights_gd()`: Batch gradient descent implementation
   - `update_weights_sgd()`: Stochastic gradient descent implementation
   - Learning rate and iteration parameters for controlling convergence

3. **Training Functions**:
   - `train_logistic_regression()`: Trains model using batch gradient descent
   - `train_logistic_regression_sgd()`: Trains model using stochastic gradient descent

4. **Model Application**:
   - Simple 2D example with visualization
   - Real-world application on ad click-through data
   - Feature preprocessing with one-hot encoding for categorical variables
   - Performance evaluation using ROC AUC score

5. **Performance Comparison**:
   - Compares execution time between gradient descent and stochastic gradient descent
   - Shows how SGD is more efficient for large datasets

The implementation provides a complete view of logistic regression, from basic principles to real-world application, and demonstrates why SGD is preferred for large-scale machine learning problems.

## Getting Started

To run these implementations, you'll need:

1. Python 3.x
2. Required packages:
   - numpy
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn
   - xgboost

Install the required packages:
```
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

Once the dependencies are installed, navigate to the respective directories and run the scripts or notebooks as described in the sections above.

## Contributing

Contributions are welcome! If you have an idea for improving the implementations or adding new algorithms, feel free to fork the repository and submit a pull request.

