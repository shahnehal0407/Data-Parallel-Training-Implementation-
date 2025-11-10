import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import time
import datetime
import matplotlib.pyplot as plt  # For plotting training results

# Default timeout duration for the process group initialization (60 seconds)
DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=60)

# Custom convolution function using torch.matmul
def custom_conv2d(input_tensor, weight_tensor, bias_tensor=None, stride=1, padding=0):
    """
    Custom implementation of 2D convolution using matrix multiplication.
    Handles padding, stride, and input/output shape management manually.
    """
    if padding > 0:
        # Apply padding to the input tensor to ensure proper dimensions after convolution
        input_tensor = F.pad(input_tensor, (padding, padding, padding, padding))
    
    batch_size, channels, height, width = input_tensor.shape  # Get input dimensions
    out_channels, _, kernel_height, kernel_width = weight_tensor.shape  # Get weight dimensions
    
    # Calculate output height and width after convolution
    output_height = (height - kernel_height) // stride + 1
    output_width = (width - kernel_width) // stride + 1

    # Unfold the input tensor to get sliding windows
    unfolded_input = input_tensor.unfold(2, kernel_height, stride).unfold(3, kernel_width, stride)
    unfolded_input = unfolded_input.contiguous().view(batch_size, channels, output_height, output_width, kernel_height * kernel_width)
    unfolded_input = unfolded_input.permute(0, 2, 3, 1, 4).reshape(batch_size, output_height * output_width, channels * kernel_height * kernel_width)

    # Flatten the weights for matrix multiplication
    flattened_weights = weight_tensor.view(out_channels, -1)
    
    # Perform matrix multiplication to compute the convolution output
    output = torch.matmul(unfolded_input, flattened_weights.t())  # Shape: (batch_size, output_height * output_width, out_channels)

    # Add the bias if provided
    if bias_tensor is not None:
        output += bias_tensor.view(1, 1, out_channels)
    
    # Reshape the output back to the expected format (batch, channels, height, width)
    output = output.view(batch_size, output_height, output_width, out_channels).permute(0, 3, 1, 2)

    return output

class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True):
        """
        Custom 2D convolutional layer with specified input and output channels,
        kernel size, stride, and padding.
        """
        super(CustomConvLayer, self).__init__()

        # Ensure kernel_size is a tuple if it's a single integer
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias for the convolutional layer
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))  # Filter weights
        if use_bias:
            self.bias = nn.Parameter(torch.randn(out_channels))  # Bias
        else:
            self.register_parameter('bias', None)

    def forward(self, input_tensor):
        """
        Performs the forward pass using custom convolution.
        """
        return custom_conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding)

class LeNet5CustomModel(nn.Module):
    def __init__(self):
        """
        LeNet-5 Model Architecture with custom convolutions.
        This follows the LeNet-5 structure for image classification.
        """
        super(LeNet5CustomModel, self).__init__()

        # Define layers of the model
        self.conv1 = CustomConvLayer(1, 6, kernel_size=5, padding=0)  # First convolutional layer (32x32 -> 28x28)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Pooling layer (28x28 -> 14x14)
        self.conv2 = CustomConvLayer(6, 16, kernel_size=5, padding=0)  # Second convolutional layer (14x14 -> 10x10)
        self.conv3 = CustomConvLayer(16, 120, kernel_size=5, padding=0)  # Third convolutional layer (10x10 -> 1x1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)  # Output layer with 10 classes for classification

    def forward(self, input_tensor):
        """
        Defines the forward pass for the LeNet-5 model.
        The input tensor goes through the convolutional and fully connected layers.
        """
        x = torch.sigmoid(self.conv1(input_tensor))  # Apply convolution + activation
        x = self.pool(x)  # Apply pooling
        x = torch.sigmoid(self.conv2(x))  # Apply second convolution + activation
        x = self.pool(x)  # Apply pooling
        x = torch.sigmoid(self.conv3(x))  # Apply third convolution + activation
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = torch.sigmoid(self.fc1(x))  # Fully connected layer with sigmoid activation
        x = self.fc2(x)  # Final output layer
        return x

def all_reduce_gradients(model, rank, world_size):
    """
    All-reduce function for synchronizing gradients across multiple ranks.
    This ensures that each rank shares and averages its gradients with others.
    """
    for name, param in model.named_parameters():
        local_gradient = param.grad.data.clone()  # Clone the gradient for the parameter

        if rank == 0:
            accumulated_gradient = local_gradient.clone()  # Start with the local gradient
            for src_rank in range(1, world_size):
                recv_tensor = torch.zeros_like(local_gradient)
                dist.recv(recv_tensor, src=src_rank)  # Receive gradients from other ranks
                accumulated_gradient += recv_tensor

            avg_gradient = accumulated_gradient / world_size  # Calculate average of gradients

            for dst_rank in range(1, world_size):
                dist.send(avg_gradient, dst=dst_rank)  # Send the averaged gradient to all ranks

            param.grad.data.copy_(avg_gradient)  # Update the gradient with the averaged value
        else:
            dist.send(local_gradient, dst=0)  # Send the local gradient to rank 0
            avg_gradient = torch.zeros_like(local_gradient)
            dist.recv(avg_gradient, src=0)  # Receive the averaged gradient from rank 0
            param.grad.data.copy_(avg_gradient)  # Update the gradient with the averaged value

def compute_accuracy(model, dataset, device):
    """
    Computes the accuracy of the model on a given dataset.
    This function is useful to evaluate the model's performance on both training and validation sets.
    """
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)  # Move data to the device (CPU)
    
    model.eval()  # Set model to evaluation mode (no gradients needed)
    with torch.no_grad():
        outputs = model(images)  # Forward pass
        _, predictions = torch.max(outputs, 1)  # Get predictions
        correct_predictions = (predictions == labels).sum().item()  # Count correct predictions

    return correct_predictions / len(labels)  # Return accuracy as a fraction of correct predictions

def train_worker(rank, world_size, backend, time_list):
    """
    Main function for each worker that trains the model in parallel across different ranks.
    It handles the data loading, training, and gradient synchronization.
    """
    start_time = time.time()  # Start measuring training time

    # Initialize the distributed process group
    store = dist.TCPStore("127.0.0.1", 29500, is_master=(rank == 0), timeout=DEFAULT_PG_TIMEOUT)
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world_size)

    device = torch.device("cpu")  # Using CPU here, could be changed to GPU if needed
    model = LeNet5CustomModel().to(device)  # Initialize the custom LeNet-5 model

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification

    transform = transforms.Compose([
        transforms.Pad(2),  # Pad image to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize based on MNIST statistics
    ])

    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    subset = Subset(full_dataset, list(range(100)))  # Use a small subset of the data for testing

    # Split dataset across ranks
    partition_size = 100 // world_size  # Number of examples per rank
    local_indices = list(range(rank * partition_size, (rank + 1) * partition_size))
    local_subset = Subset(subset, local_indices)
    loader = DataLoader(local_subset, batch_size=partition_size, shuffle=False)

    # Validation data
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    num_epochs = 100  # Number of epochs to train
    epoch_losses = []
    epoch_accuracies = []
    total_train_correct = 0
    total_train_count = 0
    total_val_correct = 0
    total_val_count = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        batch_loss = 0.0
        batch_count = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass to compute gradients

            # Synchronize gradients across all ranks
            all_reduce_gradients(model, rank, world_size)
            optimizer.step()  # Update model parameters

            batch_loss += loss.item()  # Accumulate loss
            batch_count += 1

            # Calculate accuracy on this batch
            _, predictions = torch.max(outputs, 1)
            total_train_correct += (predictions == labels).sum().item()
            total_train_count += labels.size(0)

            break  # Process only one batch per epoch for simplicity

        avg_loss = batch_loss / batch_count
        epoch_losses.append(avg_loss)

        # Calculate accuracy on the local dataset
        local_accuracy = compute_accuracy(model, local_subset, device)
        epoch_accuracies.append(local_accuracy)

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Local Accuracy = {local_accuracy*100:.2f}%")

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        total_val_correct += correct
        total_val_count += total

    if rank == 0:
        plt.figure()
        plt.plot(range(1, num_epochs+1), epoch_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch vs Loss")
        plt.legend()

        plt.figure()
        plt.plot(range(1, num_epochs+1), [a*100 for a in epoch_accuracies], label="Local Accuracy (%)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Epoch vs Local Training Accuracy")
        plt.legend()

        plt.show()

    overall_train_accuracy = total_train_correct / total_train_count
    overall_val_accuracy = total_val_correct / total_val_count

    end_time = time.time()  # Measure the end time of the training
    execution_time = end_time - start_time

    time_list.append(execution_time)  # Store the execution time for each rank

    print(f"Rank {rank}:")
    print(f"  Overall Training Accuracy: {overall_train_accuracy*100:.2f}%")
    print(f"  Overall Validation Accuracy: {overall_val_accuracy*100:.2f}%")
    print(f"  Total Execution Time: {execution_time:.2f} seconds")

    dist.barrier()  # Ensure all ranks are synchronized
    dist.destroy_process_group()  # Clean up after training is done

def main():
    world_size = 4  # Number of ranks (processes) for distributed training
    backend = "gloo"  # 'gloo' is commonly used for CPU-based training
    time_list = []  # List to collect execution times from each rank
    mp.spawn(train_worker, args=(world_size, backend, time_list), nprocs=world_size, join=True)  # Start the distributed training
    
    # Calculate the total training time across all ranks
    total_time = sum(time_list)  # Sum up the times from all ranks
    print(f"Overall Training Time: {total_time:.2f} seconds")  # Print the total training time

if __name__ == '__main__':
    main()  # Execute the training process
