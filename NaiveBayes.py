import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
class NaiveBayes:
    def __init__(self):
        self.class_priors = {}  # Prior probabilities of each class
        self.class_likelihoods = {}  # Likelihood probabilities for each class and feature
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            # Calculate prior probabilities
            self.class_priors[c] = np.sum(y == c) / len(y)

            # Calculate likelihood probabilities using non-Gaussian models (e.g., kernel density estimation)
            self.class_likelihoods[c] = {}
            for feature_idx in range(X.shape[1]):
                feature_values = X[y == c, feature_idx]
                self.class_likelihoods[c][feature_idx] = self.estimate_density(feature_values)

    def estimate_density(self, data):
        # Use a non-Gaussian density estimation method (e.g., kernel density estimation)
        hist, bin_edges = np.histogram(data, bins=10, density=True)
        density = np.zeros(len(data))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for i in range(len(data)):
            idx = np.searchsorted(bin_centers, data[i])
            if idx > 0 and idx < len(bin_centers):
                density[i] = hist[idx]
        return density

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors = {}
            for c in self.classes:
                likelihood = 1.0
                for feature_idx, feature_value in enumerate(x):
                    # Use the estimated density to compute likelihood
                    likelihood *= self.class_likelihoods[c][feature_idx][feature_value]
                posterior = self.class_priors[c] * likelihood
                posteriors[c] = posterior
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)

        return predictions

# Example usage:
# Create synthetic non-Gaussian data for demonstration
X = load_iris()['data']
y = load_iris()['target']
data = np.column_stack((X, y))


val = train_test_split(data[:,:4], data[:,4], train_size=0.8, test_size=0.2, random_state=0)

model = NaiveBayes()
model.fit(val[0],val[2])
