# models/RadialBasisFunctionModel.py

import numpy as np

from sklearn.metrics import mean_squared_error



class RBFN_model:
    def __init__(self, input_dim, num_hidden_neurons, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.num_hidden_neurons = num_hidden_neurons
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Init parameters
        self.centroids = np.random.randn(num_hidden_neurons, input_dim)
        self.betas = np.ones(num_hidden_neurons) * 0.5
        self.weights = np.random.randn(num_hidden_neurons, output_dim)

    def gaussian_rbf(self, x, center, beta):
        return np.exp(-beta * np.linalg.norm(x - center) ** 2)

    def compute_hidden_layer(self, X):
        G = np.zeros((X.shape[0], self.num_hidden_neurons))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centroids):
                G[i, j] = self.gaussian_rbf(x, center, self.betas[j])
        return G
    
    "Uses the values from train and predicts the results"
    def predict(self, X):
        G = self.compute_hidden_layer(X)
        return G @ self.weights
    
    "The function for getting the best rbf values with weights, centroids and betas"
    "if epochs not given it will as standard run 10 epochs"
    def train(self, X, y, X_val=None, y_val=None, epochs = 10):
        train_losses = []
        val_losses = []
        self.epochs = epochs
        for epoch in range(epochs):
            # Forward
            G = self.compute_hidden_layer(X)
            y_pred = G @ self.weights
            
            # Loss
            train_loss = mean_squared_error(y, y_pred)
            train_losses.append(train_loss)
            
            # Gradients
            grad_weights = -2 * G.T @ (y - y_pred) / X.shape[0]
            grad_centroids = np.zeros_like(self.centroids)
            grad_betas = np.zeros_like(self.betas)
            
            for j in range(self.num_hidden_neurons):
                diff = X - self.centroids[j]                      # shape: (N, D)
                r_sq = np.sum(diff**2, axis=1)                   # shape: (N,)
                
                # Summér (y - y_pred)*de tilsvarende weights op over output-dimensionen
                err_times_weights = np.sum((y - y_pred) * self.weights[j], axis=1)  # (N,)

                # ---------- GRADIENT WRT CENTROIDS ----------
                # Her skal du have -4 i stedet for -2
                grad_centroids[j] = (
                    -4 * self.betas[j] / X.shape[0]
                ) * np.sum(
                    err_times_weights[:, None] * G[:, j][:, None] * diff,
                    axis=0
                )

                # ---------- GRADIENT WRT BETAS ----------
                # Her er fortegnet negativt (fordi afledningen er -r^2 * g_j)
                grad_betas[j] = (
                    -2 / X.shape[0]
                ) * np.sum(
                    err_times_weights * r_sq * G[:, j]
                )
            # Parameter-updates
            self.weights -= self.learning_rate * grad_weights
            self.centroids -= self.learning_rate * grad_centroids
            self.betas -= self.learning_rate * grad_betas

            # Validation
            if X_val is not None and y_val is not None:
                G_val = self.compute_hidden_layer(X_val)
                y_val_pred = G_val @ self.weights
                val_loss = mean_squared_error(y_val, y_val_pred)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

        return train_losses, val_losses
