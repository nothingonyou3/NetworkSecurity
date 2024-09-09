# NetworkSecurity
Project implementation

The project is composed of a main.py file that can be used to run the experiments. All the following code is contained into the main.py. Depeding on the test that you want to perform you can decomment the one you need to use and comment the other 2.
To run the basic version is necessary to use this model:
autoencoder = SKSAutoencoder(input_dim=A_input_dim, hidden_dim=A_hidden_dim, k=A_k, n_of_transitional_layers=2).

For the multi-head attention test:
autoencoder = SKSAutoencoderMultiheadSA(input_dim=A_input_dim,
                                        hidden_dim=A_hidden_dim,
                                        k=A_k,
                                        num_heads=8,
                                        n_of_transitional_layers=2)

while for the residual test:
autoencoder = SKSAutoencoderResidual(input_dim=A_input_dim,
                                     hidden_dim=A_hidden_dim,
                                     k=A_k,
                                     n_of_transitional_layers=2).

The file training.py contains the command to perform the training of the project while autoencoder.py contains the basic version, multi_head_attention.py contains the multi-head attention version while the residual one is performed by residual.py.
Results.txt presents a report of the results obtained.
Classifier.py presents the logic for a binary classification.
The Excel file explains the meaning of each field in the dataset.
