import numpy as np
import torch
import os

def test(model, test_loader, criterion, device='cuda'):
    """
    Tests a PyTorch classification model.

    Args:
        model (torch.nn.Module): The neural network model to test.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        criterion (torch.nn.Module): The loss function.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model = model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    test_loss = running_loss / total
    test_accuracy = correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return test_loss, test_accuracy






def train(model, train_loader, val_loader, criterion, optimizer, save_path, num_epochs=25, patience=5,  device='cuda' ):
    """
    Trains a PyTorch classification model with validation, early stopping, and model checkpointing.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_epochs (int): Number of epochs to train the model.
        patience (int): Number of epochs to wait for improvement before stopping.
        device (str): The device to run the model on ('cuda' or 'cpu').
        save_path (str): The path to save the best model.

    Returns:
        tuple: (train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history)
    """
    model = model.to(device)
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)

                correct += (predicted == torch.max(labels, 1)[1]).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model checkpoint
            torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
            print(f'Model improved, saving to {save_path}')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history