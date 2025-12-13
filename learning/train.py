import torch
import torch.nn as nn
import torch.optim as optim

from learning.data import ResidualDataset, get_data_loaders


def train_finite_difference(
    xtraj, utraj, dt, dynamics_fn, model, lr=1e-3, epochs=100, batch_size=128, device="cpu"
):
    """Learn the perturbation function using a neural network
    Input:
        - xtraj: torch.Tensor of shape (num_samples, num_steps + 1, Nx), the state trajectory
        - utraj: torch.Tensor of shape (num_samples, num_steps, Nu), the control trajectory
        - dt: float, time step
        - dynamics: function, nominal dynamics of the system
        - model: nn.Module, the neural network model
        - lr: float, learning rate
        - epochs: int, number of epochs to train the network
        - batch_size: int, batch size
    Output:
        - model: nn.Module, the learned perturbation function
    """
    # Compute residual from data; construct data
    residual_dataset = ResidualDataset(xtraj, utraj, dynamics_fn, dt)
    model.set_mean_std(residual_dataset.residual_mean, residual_dataset.residual_std)
    train_loader, val_loader = get_data_loaders(residual_dataset, batch_size=batch_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-6)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, residual_batch in train_loader:
            x_batch, residual_batch = x_batch.to(device), residual_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            train_loss = criterion(output, residual_batch)
            train_loss.backward()
            optimizer.step()
            train_loss += train_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, residual_val in val_loader:
                x_val, residual_val = x_val.to(device), residual_val.to(device)
                output = model(x_val)
                val_loss += criterion(output, residual_val).item()
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Validation Loss: {val_loss/len(val_loader):.4f}",
            end='\r'
        )
    model.eval().to('cpu')
    return model
