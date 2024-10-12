# Pure Python Machine Learning Models

This project is where I will try to create the pure python code for making machine learning models without using any external libraries and using only any files I make or any python libraries that I make. I plan to eventually make an addon to the super app about creating custom pre-existing models and also custom new idea of models.

## Models

### Regression Models

Regression models are a type of supervised learning model that are used to predict continuous outcomes. The goal of regression is to find the best-fitting line or curve that minimizes the difference between predicted and actual values.

Types of Regression Models:

#### Simple Regression

Simple regression is a linear model that predicts a continuous outcome variable based on a single predictor variable. The goal of simple regression is to find the best-fitting line that minimizes the sum of the squared errors between predicted and actual values.

Metrics:

- Mean Squared Error (MSE): measures the average difference between predicted and actual values
- Mean Absolute Error (MAE): measures the average absolute difference between predicted and actual values
- Coefficient of Determination (R-squared): measures the proportion of the variance in the dependent variable that is predictable from the independent variable

Formulas:

- Slope: $m = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$
- Intercept: $b = \bar{y} - m\bar{x}$

#### Multiple Regression

Multiple regression is a linear model that predicts a continuous outcome variable based on multiple predictor variables. The goal of multiple regression is to find the best-fitting line or plane that minimizes the difference between predicted and actual values.

Metrics:

- Mean Squared Error (MSE): measures the average difference between predicted and actual values
- Mean Absolute Error (MAE): measures the average absolute difference between predicted and actual values
- Coefficient of Determination (R-squared): measures the proportion of the variance in the dependent variable that is predictable from the independent variables

Formulas:

- Slope: $m = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$
- Intercept: $b = \bar{y} - m\bar{x}$

#### Linear Regression

Linear regression is a linear model that predicts a continuous outcome variable based on a single predictor variable. The goal of linear regression is to find the best-fitting line that minimizes the sum of the squared errors between predicted and actual values.

Metrics:

- Mean Squared Error (MSE): measures the average difference between predicted and actual values
- Mean Absolute Error (MAE): measures the average absolute difference between predicted and actual values
- Coefficient of Determination (R-squared): measures the proportion of the variance in the dependent variable that is predictable from the independent variable

Formulas:

- Slope: $m = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$
- Intercept: $b = \bar{y} - m\bar{x}$

#### Multiple Linear Regression

Multiple linear regression is a linear model that predicts a continuous outcome variable based on multiple predictor variables. The goal of multiple linear regression is to find the best-fitting line or plane that minimizes the difference between predicted and actual values.

Metrics:

- Mean Squared Error (MSE): measures the average difference between predicted and actual values
- Mean Absolute Error (MAE): measures the average absolute difference between predicted and actual values
- Coefficient of Determination (R-squared): measures the proportion of the variance in the dependent variable that is predictable from the independent variables

Formulas:

- Slope: $m = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$
- Intercept: $b = \bar{y} - m\bar{x}$

#### Ridge Regression

Ridge regression is a linear model that predicts a continuous outcome variable based on multiple predictor variables. The goal of ridge regression is to find the best-fitting line or plane that minimizes the difference between predicted and actual values, while also penalizing the model for large coefficients.

Metrics:

- Mean Squared Error (MSE): measures the average difference between predicted and actual values
- Mean Absolute Error (MAE): measures the average absolute difference between predicted and actual values
- Coefficient of Determination (R-squared): measures the proportion of the variance in the dependent variable that is predictable from the independent variables

Formulas:

- MSE: SSE / n
- MAE: |y - y'| / n
- R-squared: 1 - (SSE / SST)

where SSE is the sum of the squared errors, SST is the total sum of squares, n is the sample size, y is the dependent variable, and y' is the predicted value.

#### Lasso Regression

Lasso regression is a linear model that predicts a continuous outcome variable based on multiple predictor variables. The goal of lasso regression is to find the best-fitting line or plane that minimizes the difference between predicted and actual values, while also selecting the most important features.

Metrics:

- Mean Squared Error (MSE): measures the average difference between predicted and actual values
- Mean Absolute Error (MAE): measures the average absolute difference between predicted and actual values
- Coefficient of Determination (R-squared): measures the proportion of the variance in the dependent variable that is predictable from the independent variables

Formulas:

- MSE: SSE / n
- MAE: |y - y'| / n
- R-squared: 1 - (SSE / SST)

where SSE is the sum of the squared errors, SST is the total sum of squares, n is the sample size, y is the dependent variable, and y' is the predicted value.

Background Math:

- Linear Algebra: matrix operations, vector spaces, eigenvalues and eigenvectors
- Calculus: optimization techniques, derivatives, integrals
- Probability Theory: probability distributions, Bayes' theorem

### Classification Models

Classification models are a type of supervised learning model that are used to predict categorical outcomes. The goal of classification is to find the best-fitting decision boundary that minimizes the difference between predicted and actual values.

Types of Classification Models:

#### Logistic Regression

Logistic regression is a linear model that predicts a categorical outcome variable based on multiple predictor variables. The goal of logistic regression is to find the best-fitting decision boundary that minimizes the difference between predicted and actual values.

Metrics:

- Accuracy: measures the proportion of correctly classified instances
- Precision: measures the proportion of true positives among all positive predictions
- Recall: measures the proportion of true positives among all actual positive instances
- F1 Score: measures the harmonic mean of precision and recall

Formulas:

- Slope: $m = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$
- Intercept: $b = \bar{y} - m\bar{x}$

#### Support Vector Machines

Support vector machines is a linear model that predicts a categorical outcome variable based on multiple predictor variables. The goal of support vector machines is to find the best-fitting decision boundary that minimizes the difference between predicted and actual values.

Metrics:

- Accuracy: measures the proportion of correctly classified instances
- Precision: measures the proportion of true positives among all positive predictions
- Recall: measures the proportion of true positives among all actual positive instances
- F1 Score: measures the harmonic mean of precision and recall

Formulas:

- SVM: $y = \sum_{i=1}^n \alpha_i x_i + b$

#### Tree-Based Models

##### Decision Tree

Decision tree is a non-parametric model that predicts a categorical outcome variable based on multiple predictor variables. The goal of decision tree is to find the best-fitting decision boundary that minimizes the difference between predicted and actual values.

Metrics:

- Accuracy: measures the proportion of correctly classified instances
- Precision: measures the proportion of true positives among all positive predictions
- Recall: measures the proportion of true positives among all actual positive instances
- F1 Score: measures the harmonic mean of precision and recall

Formulas:

- Entropy: $H(X) = -\sum_{x \in X} p(x) \log_2 p(x)$
- Information Gain: $IG(X, Y) = H(X) - H(X|Y)$

##### Random Forest

Random forest is an ensemble of decision trees that predicts a categorical outcome variable based on multiple predictor variables. The goal of random forest is to find the best-fitting decision boundary that minimizes the difference between predicted and actual values.

Metrics:

- Accuracy: measures the proportion of correctly classified instances
- Precision: measures the proportion of true positives among all positive predictions
- Recall: measures the proportion of true positives among all actual positive instances
- F1 Score: measures the harmonic mean of precision and recall

Formulas:

- Entropy: $H(X) = -\sum_{x \in X} p(x) \log_2 p(x)$
- Information Gain: $IG(X, Y) = H(X) - H(X|Y)$

### Clustering Models

Clustering models are a type of unsupervised learning model that are used to predict categorical outcomes. The goal of clustering models is to find the best-fitting clusters that minimize the difference between predicted and actual values.

Types of Clustering Models:

#### K-Means Clustering

K-means clustering is a type of non-parametric model that predicts a categorical outcome variable based on multiple predictor variables. The goal of K-means clustering is to find the best-fitting clusters that minimize the difference between predicted and actual values.

Metrics:

- Silhouette Coefficient: measures the separation and cohesion of each cluster
- Calinski-Harabasz Index: measures the ratio of between-cluster variance to within-cluster variance

Formulas:

- Silhouette Coefficient: $S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
- Calinski-Harabasz Index: $CH = \frac{tr(B) \cdot (N - k)}{tr(W) \cdot (k - 1)}$

#### Hierarchical Clustering

Hierarchical clustering is an ensemble of K-means clustering models that are used to predict a categorical outcome variable based on multiple predictor variables. The goal of hierarchical clustering is to find the best-fitting clusters that minimize the difference between predicted and actual values.

Metrics:

- Silhouette Coefficient: measures the separation and cohesion of each cluster
- Calinski-Harabasz Index: measures the ratio of between-cluster variance to within-cluster variance

Formulas:

- Silhouette Coefficient: $S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
- Calinski-Harabasz Index: $CH = \frac{tr(B) \cdot (N - k)}{tr(W) \cdot (k - 1)}$
