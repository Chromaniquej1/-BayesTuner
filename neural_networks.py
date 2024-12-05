import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout, neuronPct, neuronShrink):
        super(NeuralNetwork, self).__init__()

        layers = []
        neuronCount = int(neuronPct * 5000)
        layer = 0
        prev_count = input_dim

        while neuronCount > 25 and layer < 10:
            layers.append(nn.Linear(prev_count, neuronCount))
            prev_count = neuronCount
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))
            neuronCount = int(neuronCount * neuronShrink)
            layer += 1

        layers.append(nn.Linear(prev_count, 1))  # Assuming binary classification (adjust if multi-class)
        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
