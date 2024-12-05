import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import statistics
from sklearn import metrics

def evaluate_network(x_tensor, y_tensor, input_dim, dropout=0.2, neuronPct=0.1, neuronShrink=0.25, learning_rate=1e-3, EPOCHS=500, SPLITS=2, PATIENCE=10):
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.1)
    mean_benchmark = []

    for train, test in boot.split(x_tensor, np.argmax(y_tensor, axis=1)):
        x_train = x_tensor[train]
        y_train = y_tensor[train]
        x_test = x_tensor[test]
        y_test = y_tensor[test]

        model = NeuralNetwork(input_dim, dropout, neuronPct, neuronShrink).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        dataset_train = TensorDataset(x_train, y_train)
        loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            for batch_x, batch_y in loader_train:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs_test = model(x_test)
                val_loss = criterion(outputs_test, y_test).item()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        # Evaluate
        with torch.no_grad():
            model.eval()
            pred = model(x_test).cpu().numpy()
            y_compare = y_test.cpu().numpy()
            score = metrics.log_loss(y_compare, pred)
            mean_benchmark.append(score)

    return -statistics.mean(mean_benchmark)
