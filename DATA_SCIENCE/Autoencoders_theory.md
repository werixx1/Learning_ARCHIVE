<h3 align="center">Autoencoders</h3>

  <p align="center">
    Simple explanation on autoencoders and example of usage
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>


## Table of contents

- [Explanation](#explanation)
- [Usage](#usage)
- [Code implementation](#code-implementation)


## Explanation
> Key words: dimensionality reduction, image reconstruction, anomaly detection

**Definition**
Autoencoders learn efficient **representations of data**, primarily for the purpose of dimensionality reduction, feature extraction and noise removal.
- autoencoders are trained to discover latent variables of the input data: hidden or random variables that, despite not being directly observable, fundamentally **inform the way data is distributed**
- use unsupervised learning, model trains by minimizing reconstruction error using loss functions like **MSE** (ex. for image - how similar a reconstructed image is to the original, average distance of difference of each pixel)
- they consist of two parts: 
   - **ENCODER**: takes input (data, image etc), and **learns** how to efficiently compress and decode data while retaining its most important features (ex. feature extraction in convolution to reconstruct a clean image from noisy one), captures important features by reducing dimensionality 
    - **DECODER**: learns how to deconstruct that data compression to the original (data/image) structure as close as it can get, rebuilds the data from its compressed representation
- lower-dimensional representation of data that captures its essential features and underlying patterns is called **LATENT SPACE**

**Structure**
 ```input -> ENCODER -> CODE (bottleneck, most compressed part) -> DECODER -> output```
![](/resources/imgs/Autoencoder.png)
 - autoencoders work so great in **anomaly detection** because they can easly tell when something doesn't fit their learned status quo in data (since it learns most important patterns to reconstruct data from it) 
 - can give better results than PCA for reducing dimensionality on data that is not linear, since it captures complex **non-linear correlations** 


## Usage
`Image segmentation` `Feature extraction` `Noise removal` `Error detection` 

## Code implementation
`source: Geeksforgeeks`
[ reconstruction of grayscale images ]
> Tensorflow
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.datasets import mnist
```
```python
class SimpleAutoencoder(Model):
    def __init__(self, latent_dimensions):
        super(SimpleAutoencoder, self).__init__()

        # squeeze
        self.encoder = tf.keras.Sequential([
            # input layer expecting grayscale images of size 28x28
            layers.Input(shape=(28, 28, 1)), 
            layers.Flatten(),
            # dense layer that compresses the input to the latent space
            # using ReLU activation
            layers.Dense(latent_dimensions, activation='relu'),
        ])
        
        # unsqueeze
        self.decoder = tf.keras.Sequential([
            # dense layer that expands the latent vector back to the original 
            # image size with sigmoid activation
            layers.Dense(28 * 28, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ])
    
    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded
```
--- 
> PyTorch
```python
import torch
from torch import nn, optim
```
```python
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            # each layer has less and less nodes to compress data
            nn.Linear(28 * 28, 128),
            nn.ReLU(), # <- relu helps with reducing dimensions
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9)
        )
        self.decoder = nn.Sequential(
            # each layer has more nodes to decompress it :)
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid() # <- ends with sigmoid func to output pixel values
                         # between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

