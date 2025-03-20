# models/RBFKeras.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.cluster import KMeans

def initialize_centroids(X, num_neurons):
    """Bruger KMeans til at få meningsfulde centroids i stedet for random."""
    kmeans = KMeans(n_clusters=num_neurons, n_init='auto').fit(X)
    return kmeans.cluster_centers_

class RBFLayer(layers.Layer):
    def __init__(self, num_neurons, betas=0.5, initial_centroids=None, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.num_neurons = num_neurons
        self.betas_init = betas
        self.initial_centroids = initial_centroids

    def build(self, input_shape):
        if self.initial_centroids is not None:
            # Centroids init med KMeans
            centroid_init = tf.keras.initializers.Constant(self.initial_centroids)
        else:
            centroid_init = 'random_normal'

        self.centroids = self.add_weight(
            name='centroids',
            shape=(self.num_neurons, input_shape[-1]),
            initializer=centroid_init,
            trainable=True
        )
        self.betas = self.add_weight(
            name='betas',
            shape=(self.num_neurons,),
            initializer=tf.keras.initializers.Constant(self.betas_init),
            trainable=True
        )

    def call(self, inputs):
        # inputs shape: (batch_size, input_dim)
        # centroids shape: (num_neurons, input_dim)
        # => diff shape: (batch_size, num_neurons, input_dim)
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.centroids, axis=0)
        squared_dist = tf.reduce_sum(tf.square(diff), axis=-1)  # (batch_size, num_neurons)
        # output shape: (batch_size, num_neurons)
        return tf.exp(-self.betas * squared_dist)
    
    def get_config(self):
        config = super(RBFLayer, self).get_config()
        config.update({
            "num_neurons": self.num_neurons,
            "betas": self.betas_init,
            # Hvis initial_centroids er en NumPy-array, skal du konvertere den til en liste
            "initial_centroids": self.initial_centroids.tolist() if self.initial_centroids is not None else None,
        })
        return config

def build_rbf_model(input_dim, num_hidden_neurons, output_dim, initial_centroids=None):
    """Byg en Keras Sequential‐model med RBFLayer + Dense output."""
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(RBFLayer(num_neurons=num_hidden_neurons,
                       betas=0.5,
                       initial_centroids=initial_centroids))
    model.add(layers.Dense(output_dim))  # lineært output‐lag
    return model
