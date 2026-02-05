# Import libraries
from typing import Tuple
from torch import nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader, TensorDataset  
import numpy as np
import matplotlib.pyplot as plt


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Encoder. Hidden dimension and latent dimensions are the same. 
class Encoder(nn.Module):
    """
    The Encoder network. 
    A deep neural network that learns a lower-dimensional representation of the input data by mapping it into an embedding.
    """
    def __init__(self, input_dim: int, 
                 hidden_layers: Tuple[int] = [500, 500, 2000, 10],
                 dropout_rate: float = 0.2,
                 acitvation = nn.ReLU()
                ):
        
        super().__init__()

        # First layer, the input layer
        self.input_layer = torch.nn.Linear(input_dim, hidden_layers[0])
        self.n_layers = 0

        for i in range(0, len(hidden_layers) - 1):
            setattr(self, f'dense_{i}', torch.nn.Linear(hidden_layers[i],
                                                                 hidden_layers[i+1])
                    )
            self.n_layers += 1
        
        self.activation = acitvation
        self.hidden_layers = hidden_layers

        # Add dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
    
    def forward(self, x: Tensor) -> Tensor:
        # Special treatment for the input layer
        x = self.activation(self.input_layer(x))

        for i in range(0, self.n_layers-1):
            x = self.activation(getattr(self, f'dense_{i}')(x))
            x = self.dropout(x)
        
        output_layer = getattr(self, f'dense_{self.n_layers-1}')(x)
        return output_layer
    
# Define the Decoder
class Decoder(nn.Module):
    """
    Same as the encoder, but the layers are in reverse order. 
    So, we pass the encoder as input and use its hidden_sizes to specify the decoder network.
    """
    def __init__(self,
                 encoder,
                 activation=nn.ReLU()
                ):
        super().__init__()
        self.hidden_layers = encoder.hidden_layers
        n_layers = encoder.n_layers
        self.hidden_layers = self.hidden_layers[::-1]
        
        # Reversed order -> dense_0 will be the first to apply here
        for i in range(0, n_layers):
            setattr(self, f"dense_{i}", torch.nn.Linear(self.hidden_layers[i],
                                                        self.hidden_layers[i+1])
                   )
        self.output_layer = torch.nn.Linear(self.hidden_layers[-1],
                                                        encoder.input_dim)
        self.n_layers = n_layers
        self.activation = activation
        self.dropout  = nn.Dropout(encoder.dropout_rate)

        
    def forward(self, x:Tensor) -> Tensor:
        for i in range(0, self.n_layers):
            dense_i = getattr(self, f"dense_{i}")
            x = dense_i(x)
            x = self.activation(x)
            x = self.dropout(x)
        return self.output_layer(x)
    
# Combine Encoder and Decoder into the AutoEncoder
class AutoEncoder(nn.Module):
    '''
    The Autoencoder network, consisting of an encoder and a decoder.
    1. The encoder maps the input data to a lower-dimensional embedding.
    2. The decoder reconstructs the input data from the embedding.
    3. The network is trained to minimize the reconstruction error between the input and the output
    4. This forces the encoder to learn a compressed representation of the data that captures its most important features.
    5. The architecture of the encoder and decoder can be customized by specifying the number and size of hidden layers.
    6. Dropout can be applied to prevent overfitting.
    7. The forward method returns both the encoded representation and the reconstructed output.
    8. This implementation uses PyTorch and is designed for flexibility and ease of use in various applications.
    '''
    def __init__(self, input_dim: int, 
                 hidden_layers: Tuple[int] = [500, 500, 2000, 10],
                 dropout_rate: float = 0.2
                 ):

        super().__init__()
        self.encoder = Encoder(input_dim, hidden_layers, dropout_rate)
        
        # Decoder uses the same architecture as the encoder but in reverse
        self.decoder = Decoder(self.encoder)
        self.hidden_layers = hidden_layers

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
        

       