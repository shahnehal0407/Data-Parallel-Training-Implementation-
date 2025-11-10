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

# Default timeout duration for process group initialization (60 seconds)
DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=60)

# Custom convolution operation implemented using torch.matmul
def custom_conv2d(input_tensor, weight_tensor, bias_tensor=None, stride=1, padding=0):
    """
    A custom 2D convolution function using matrix multiplication (torch.matmul).
    Handles stride, padding, and reshaping the input tensor for efficient computation.
    """
    if padding > 0:
        # Padding the input tensor to ensure the output dimensions are as expected
        input_tensor = F.pad(input_tensor, (padding, padding, padding, padding))
    
    batch_size, num_channels, height, width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight_tensor.shape
    
    # Calculate the output dimensions after the convolution operation
    output_height = (height - kernel_height) // stride + 1
    output_width = (width - kernel_width) // stride + 1

    # Unfolding the input tensor to get sliding windows for the convolution
    unfolded_input = input_tensor.unfold(2, kernel_height, stride).unfold(3, kernel_width, stride)
    unfolded_input = unfolded_input.contiguous().view(batch_size, num_channels, output_height, output_width, kernel_height * kernel_width)
    unfolded_input = unfolded_input.permute(0, 2, 3, 1, 4).reshape(batch_size, output_height * output_width, num_channels * kernel_height * kernel_width)

    # Flatten the weights for matrix multiplication
    flattened_weights = weight_tensor.view(out_channels, -1)
    
    # Perform matrix multiplication between the unfolded input and the flattened weights
    output = torch.matmul(unfolded_input, flattened_weights.t())  # Shape: (batch_size, output_height * output_width, out_channels)

    # Add the bias to the result if provided
    if bias_tensor is not None:
        output += bias_tensor.view(1, 1, out_channels)
    
    # Reshape the output back to the (batch_size, out_channels, output_height, output_width) format
    output = output.view(batch_size, output_height, output_width, out_channels).permute(0, 3, 1, 2)

    return output

# Custom 2D convolutional layer using our custom convolution function
class CustomConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True):
        """
        Custom Convolutional Layer class using the custom 2D convolution operation.
        Initializes weights and bias (if required).
        """
        super(CustomConv2dLayer, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)  # Ensure kernel_size is a tuple if it's an integer
        
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias (if needed)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))  # Filter weights
        if use_bias:
            self.bias = nn.Parameter(torch.randn(out_channels))  # Bias term
        else:
            self.register_parameter('bias', None)

    def forward(self, input_tensor):
        """
        The forward pass applies the custom convolution operation.
        """
        return custom_conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding)

# LeNet-5 custom model with custom convolution layers
class LeNet5CustomModel(nn.Module):
    def __init__(self):
        """
        Defines the custom LeNet-5 model architecture using custom convolution layers.
        This model is designed for image classification (e.g., MNIST dataset).
        """
        super(LeNet5CustomModel, self).__init__()

        # Define layers: Convolutional layers followed by fully connected layers
        self.conv1 = CustomConv2dLayer(1, 6, kernel_size=5, padding=0)  # First convolution (28x28 -> 24x24)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Pooling layer (24x24 -> 12x12)
        self.conv2 = CustomConv2dLayer(6, 16, kernel_size=5, padding=0)  # Second convolution (12x12 -> 8x8)
        self.conv3 = CustomConv2dLayer(16, 120, kernel_size=5, padding=0)  # Third convolution (8x8 -> 4x4)

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)  # Fully connected layer (flattened size 120)
        self.fc2 = nn.Linear(84, 10)  # Output layer (10 classes for MNIST)

    def forward(self, input_tensor):
        """
        Defines the forward pass for the model.
        The input tensor passes through convolutional layers and fully connected layers.
        """
        x = torch.sigmoid(self.conv1(input_tensor))  # Apply convolution + activation
        x = self.pool(x)  # Apply pooling
        x = torch.sigmoid(self.conv2(x))  # Apply second convolution + activation
        x = self.pool(x)  # Apply pooling
        x = torch.sigmoid(self.conv3(x))  # Apply third convolution + activation
        x = x.view(x.size(0), -1)  # Flatten the output to pass into fully connected layers
        x = torch.sigmoid(self.fc1(x))  # Apply first fully connected layer
        x = self.fc2(x)  # Output layer (logits for classification)
        return x

# Ring All-Reduce function for gradient synchronization across ranks
def ring_all_reduce(tensor, rank, world_size):
    """
    Ring-based All-Reduce for gradient synchronization.
    This function allows all ranks to share and synchronize their gradients during training.
    """
    transmit_buffer = tensor.clone()
    received_buffer = torch.zeros_like(tensor)
    
    # Perform ring all-reduce (each rank sends and receives gradients in a ring fashion)
    for _ in range(world_size - 1):
        transmit_rank = (rank + 1) % world_size
        receive_rank = (rank - 1 + world_size) % world_size
        
        # Send and receive gradients across ranks
        send_req = dist.isend(tensor=transmit_buffer, dst=transmit_rank)
        recv_req = dist.irecv(tensor=received_buffer, src=receive_rank)
        
        send_req.wait()  # Wait for send to complete
        recv_req.wait()  # Wait for receive to complete
        
        tensor.add_(received_buffer)  # Update the gradients with the received gradients
        transmit_buffer.copy_(received_buffer)  # Update the transmit buffer for the next iteration
    
    # Average the gradients across all ranks
    tensor.div_(world_size)

# Helper function to compute local accuracy on a dataset
def compute_local_accuracy(model, dataset, device):
    """
    Computes the accuracy of the model on a local dataset.
    This function is used to compute accuracy on the training subset.
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)  # Move to device (CPU in this case)
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(images)  # Forward pass
        _, predictions = torch.max(outputs, 1)  # Get predictions from model output
        correct_predictions = (predictions == labels).sum().item()  # Count correct predictions

    return correct_predictions / len(labels)  # Return accuracy as fraction

# Main worker function for distributed training
def main_worker(rank, world_size, backend):
    """
    The main training loop for each rank in a distributed setup.
    This function trains the model, synchronizes gradients, and computes accuracy.
    """
    start_time = time.time()  # Start timing the training process

    store = dist.TCPStore("127.0.0.1", 29500, is_master=(rank == 0), timeout=DEFAULT_PG_TIMEOUT)
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world_size)

    device = torch.device("cpu")  # Use CPU for training (could be modified for GPU)
    model = LeNet5CustomModel().to(device)  # Initialize the custom model

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()  # Cross entropy loss for classification

    transform = transforms.Compose([
        transforms.Pad(2),  # Pad input to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize based on MNIST statistics
    ])

    # Load MNIST dataset and use a small subset of 100 examples for distributed training
    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    subset = Subset(full_dataset, list(range(100)))  # Use only 100 examples for faster training

    partition_size = 100 // world_size  # Split data into equal parts for each rank
    local_indices = list(range(rank * partition_size, (rank + 1) * partition_size))
    local_subset = Subset(subset, local_indices)
    loader = DataLoader(local_subset, batch_size=partition_size, shuffle=False)

    # Validation dataset (using the test set)
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

    num_epochs = 100  # Number of epochs for training
    epoch_losses = []  # List to store the loss for each epoch
    epoch_accuracies = []  # List to store accuracy for each epoch
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
            outputs = model(images)  # Perform forward pass
            loss = loss_fn(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass to compute gradients

            # Apply ring all-reduce to synchronize gradients
            for param in model.parameters():
                if param.grad is not None:
                    ring_all_reduce(param.grad.data, rank, world_size)

            optimizer.step()  # Update model parameters based on gradients

            batch_loss += loss.item()
            batch_count += 1

            # Compute accuracy for this batch
            _, predictions = torch.max(outputs, 1)
            total_train_correct += (predictions == labels).sum().item()
            total_train_count += labels.size(0)

            break  # Only process one batch per epoch for simplicity

        avg_loss = batch_loss / batch_count
        epoch_losses.append(avg_loss)

        # Compute local accuracy
        local_accuracy = compute_local_accuracy(model, local_subset, device)
        epoch_accuracies.append(local_accuracy)

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Local Accuracy = {local_accuracy*100:.2f}%")

        # Calculate validation accuracy
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

    # After training, plot training loss and accuracy
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

    # Calculate overall accuracy for training and validation
    overall_train_accuracy = total_train_correct / total_train_count
    overall_val_accuracy = total_val_correct / total_val_count

    end_time = time.time()  # End time for the training
    execution_time = end_time - start_time

    print(f"Rank {rank}:")
    print(f"  Overall Training Accuracy: {overall_train_accuracy*100:.2f}%")
    print(f"  Overall Validation Accuracy: {overall_val_accuracy*100:.2f}%")
    print(f"  Total Training Time: {execution_time:.2f} seconds")  # Print the training time

    dist.barrier()  # Ensure all ranks are synchronized
    dist.destroy_process_group()  # Clean up after training

def main():
    world_size = 4  # Number of ranks (processes) for distributed training
    backend = "gloo"  # 'gloo' is a backend commonly used for CPU-based training
    mp.spawn(main_worker, args=(world_size, backend), nprocs=world_size, join=True)  # Start distributed training

if __name__ == '__main__':
    main()  # Run the training process
