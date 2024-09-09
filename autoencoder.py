import torch
import torch.nn as nn


# Custom K-Sparse Layer
class KSparse(nn.Module):
    def __init__(self, k):
        super(KSparse, self).__init__()
        self.k = k

    def forward(self, inputs):
        top_k_values, _ = torch.topk(inputs, self.k, dim=1)
        min_top_k = torch.min(top_k_values, dim=1, keepdim=True).values
        sparse_output = torch.where(inputs >= min_top_k, inputs, torch.zeros_like(inputs))
        return sparse_output


# Custom Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
        self.W_Q = nn.Linear(units, units)
        self.W_K = nn.Linear(units, units)
        self.W_V = nn.Linear(units, units)
        self.scale = torch.sqrt(torch.FloatTensor([units]))

    def forward(self, inputs):
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)

        # Score calculation (Scaled Dot-Product Attention)
        score = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(score, dim=-1)

        # Weighted sum of values
        context = torch.matmul(attention_weights, V)
        return context


# Define the Self-Attention K-Sparse Autoencoder Model
class SKSAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, n_of_transitional_layers=3, return_sparsity=False):
        super(SKSAutoencoder, self).__init__()
        layers_dims = self.__class__.__generate_intermediate_dims(input_dim=input_dim,
                                                                  hidden_dim=hidden_dim,
                                                                  num_layers=n_of_transitional_layers)

        # We create the encoder
        self.encoder = self.__class__.__create_encoder(input_dim=input_dim, intermediate_dims=layers_dims)

        self.attention = SelfAttention(hidden_dim)
        self.k_sparse = KSparse(k)

        # We create the decoder
        self.decoder = self.__class__.__create_decoder(output_dim=input_dim, intermediate_dims=layers_dims)
        self._out_dim = None

        self.return_sparsity = return_sparsity  # To use the composite loss function if desired

    @staticmethod
    def __generate_intermediate_dims(input_dim: int, hidden_dim: int, num_layers: int) -> list:
        """
        Generate intermediate layer dimensions in descending order from input_dim to hidden_dim.
        """
        # Calculate the step size
        step_size = (input_dim - hidden_dim) // num_layers

        # Generate intermediate dimensions
        intermediate_dims = [input_dim - i * step_size for i in range(1, num_layers)]
        intermediate_dims.append(hidden_dim)

        return intermediate_dims

    @staticmethod
    def __create_encoder(input_dim: int, intermediate_dims: list) -> nn.Sequential:
        """
        Create a sequence of linear layers that reduces the dimension from start_dim to each of the dimensions
        in intermediate_dims.
        """
        layers = []
        current_dim = input_dim

        for next_dim in intermediate_dims:
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())  # Add a non-linearity (ReLU) after each layer
            current_dim = next_dim

        return nn.Sequential(*layers)

    @staticmethod
    def __create_decoder(output_dim: int, intermediate_dims: list) -> nn.Sequential:
        """
        Create a sequence of linear layers that increases the dimension from start_dim to each of the dimensions
        in intermediate_dims.
        """
        # The code may seem a bit complex though we're just doing the opposite of __create_encoder, nothing more
        layers = []

        current_dim = intermediate_dims[-1]  # We start with the hidden dimension, - the last one in the list -
        del(intermediate_dims[-1])

        sorted_intermediate_dims = sorted(intermediate_dims)  # We need to order this in ascending order
        sorted_intermediate_dims.append(output_dim)  # And we need to add the output dimension, which wasn't included

        for i, next_dim in enumerate(sorted_intermediate_dims):
            if i + 1 < len(sorted_intermediate_dims):
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(nn.ReLU())
                current_dim = next_dim
            else:  # The last layer shall be a sigmoid
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, inputs):
        z = self.encode(inputs)  # Encoder
        reconstructed = self.decode(z)  # Decoder
        if self.return_sparsity:
            return reconstructed, z
        return reconstructed

    def encode(self, x):
        self.__set_decoder_dimension(x.size())  # This is necessary to the decoder
        x = x.view(x.size()[0], -1)  # flattening the tensor (necessary for fully connected layers)

        x = self.encoder(x)
        x = self.attention(x)
        x = self.k_sparse(x)
        return x

    def __set_decoder_dimension(self, dimensions):
        self._out_dim = dimensions

    def decode(self, x):
        x = self.decoder(x)
        x = x.view(self._out_dim)
        return x
