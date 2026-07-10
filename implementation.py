import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTradingModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleTradingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

def generate_adversarial_samples(model, data, target, epsilon=0.01):
    data.requires_grad = True
    output = model(data)
    loss = nn.MSELoss()(output, target)
    model.zero_grad()
    loss.backward()
    adversarial_data = data + epsilon * data.grad.sign()
    return adversarial_data.detach()

def train_model(model, data, target, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate_model(model, data, target):
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss = nn.MSELoss()(output, target)
    return loss.item()

if __name__ == '__main__':
    # Dummy data
    np.random.seed(42)
    torch.manual_seed(42)
    input_dim = 10
    num_samples = 100

    # Generate synthetic data
    X = np.random.rand(num_samples, input_dim).astype(np.float32)
    y = (np.sum(X, axis=1) + np.random.normal(0, 0.1, num_samples)).astype(np.float32).reshape(-1, 1)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Initialize model
    model = SimpleTradingModel(input_dim)

    # Train model
    train_model(model, X_tensor, y_tensor, epochs=200, lr=0.01)

    # Evaluate model on clean data
    clean_loss = evaluate_model(model, X_tensor, y_tensor)
    print(f"Loss on clean data: {clean_loss}")

    # Generate adversarial samples
    adversarial_X = generate_adversarial_samples(model, X_tensor, y_tensor, epsilon=0.1)

    # Evaluate model on adversarial data
    adversarial_loss = evaluate_model(model, adversarial_X, y_tensor)
    print(f"Loss on adversarial data: {adversarial_loss}")