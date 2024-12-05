import torch
from data_preprocessing import load_and_preprocess_data
from neural_network import NeuralNetwork
from training import evaluate_network
from hyperparameter_optimization import bayesian_optimization

# Startup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Data Preprocessing
x, y = load_and_preprocess_data()

# Convert data to PyTorch tensors
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.LongTensor(np.argmax(y, axis=1)).to(device)

# Hyperparameter Optimization
bayesian_optimization(evaluate_network)
