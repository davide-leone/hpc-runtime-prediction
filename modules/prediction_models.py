import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class NAGPolynomialRegressor:
    '''
    
    This is the actual on-line implementation from the paper: 
    “Improving Backfilling by using Machine Learning to Predict Running Times”
    
    '''
    def __init__(self, degree=2, alpha=0.01, eta=1e-8, gamma=0.9):
        self.degree = degree
        self.alpha = alpha  # L2 regularization strength
        self.eta = eta  # Learning rate
        self.gamma = gamma  # NAG momentum factor
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.w = None  # Weight vector
        self.v = None  # Velocity for NAG updates

    def transform_features(self, X):
        X_poly = self.poly.fit_transform(X)
        return self.scaler.fit_transform(X_poly)  # Standardize features

    def fit(self, X, y):
        X_poly = self.transform_features(X)
        n_samples, n_features = X_poly.shape

        if self.w is None:
            self.w = np.zeros(n_features)
            self.v = np.zeros(n_features)

        for i in range(n_samples):
            xi, yi = X_poly[i], y[i]
            
            # Compute Lookahead Gradient (NAG step)
            w_ahead = self.w - self.gamma * self.v  # Lookahead step
            gradient = -2 * (yi - np.dot(xi, w_ahead)) * xi + 2 * self.alpha * w_ahead
            
            # Clip extreme values to prevent instability
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Update velocity and weights
            self.v = self.gamma * self.v + self.eta * gradient
            self.w -= self.v

    def predict(self, X):
        X_poly = self.scaler.transform(self.poly.transform(X))
        return np.maximum(np.dot(X_poly, self.w), 1).astype(int)  # Ensure non-negative predictions


class RidgePolynomialRegressor:
    '''

    Since we have all the data available we can use an off-line implementation which uses Ridge regression
    
    '''
    def __init__(self, degree=2, alpha=0.01):
        self.degree = degree
        self.alpha = alpha  # L2 regularization strength
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)  # Standardize features
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)  # Standardize using the same scaler
        partial = self.model.predict(X_scaled)

        # Ensure predictions stay in valid range (0 to 86400 seconds)
        result = np.where(partial < 1, 1, np.where(partial > 86400, 86400, partial)).astype(int)
        
        return result  # Returns an array of integers