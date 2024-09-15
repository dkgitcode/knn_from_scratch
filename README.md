# KNN (K-Nearest Neighbors) from scratch
This is a simple implementation of the K-Nearest Neighbors algorithm from scratch in Python. In these notebooks, we use the Dry Bean dataset from the UCI ML Repository to classify bean types based on 16 features.

## Basic Theory

K-Nearest Neighbors is an algorithm that places the training data in a multi-dimensional space and classifies new data points based on how close they are to the clusters of training data. This "closeness" is determined by a distance metric, such as Euclidean distance. 

$$
\begin{equation}
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\end{equation}
$$

Using this equation, we can calculate the distance between two points $x$ and $y$ in an $n$-dimensional space. After we have "plotted" the training data in this space, we can try out a new data point and calculate the distance to every other point. The top k closest points are then used to classify the new data point.

### Classification

To decide the class of the new data point, the $k$ closest points vote by majority rule. The class with the most votes is the class assigned to the new data point. In our case, we just classified whether a bean is a Seker, Barbunya, Bombay, Cali, Dermosan, Horoz or Sira. 

## Implementation

```python
knn = k_nearest_neighbors(X_train, x_feature, 5)
votes = [i[-1] for i in knn]
majority_vote = max(set(votes), key=votes.count)
confidence = votes.count(majority_vote) / len(votes)
```

In this code snippet, we use the `k_nearest_neighbors` function to find the 5 closest points to the new data point. We then extract the votes from these points and find the majority vote. The confidence is calculated by dividing the number of votes for the majority class by the total number of votes. 

### k_nearest_neighbors in Python

```python
def k_nearest_neighbors(X, x1, k=5):
    distances = []
    y_values = X[:,-1]
    for i in range(len(X)):
        d = distance(X[i][:-1], x1)
        distances.append((d, y_values[i]))
    distances.sort()
    return distances[:k]
```

## Results

In order to grade our classifier, we use an accuracy score to determine how well it performs. The accuracy score is calculated by dividing the number of correct predictions by the total number of predictions. We also run multiple iterations of the algorithm on shuffled data to get a more accurate accuracy (pun intended) score.

### Comparison to Scikit-Learn

Scikit learn's implementation allows for multi threading, different distance metrics, and other optimizations that make it faster, and potentially more accurate. 

However, as far as results, our implementation is very close to the scikit-learn implementation.