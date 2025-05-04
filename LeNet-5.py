# ---- Dependencies
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# Set random seed for reproducibility across runs
torch.manual_seed(42)

# ---- Dataset loading and preprocessing
# Import and prepare dataset and dataloader
# "Reduced MNIST dataset" is not a standard dataset.
# For portability, full MNIST dataset is loaded and then reduced to be "Reduced MNIST".
# Imported from torchvision

# Apply transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data as normalization will guarantee N(0,1) distribution
])

# Get dataset and dataloader
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


def reduce_mnist(dataset, n, random_seed=42):
    # Function that reduces MNIST dataset to have "n" samples per digit randomly selected
    rng = np.random.RandomState(random_seed)  # Setting random state for reproducibility
    indices = []  # Initializing empty list to store selected indices
    for digit in range(10):  # Looping over 10 digits 0->9
        digit_indices = np.where(dataset.targets.numpy() == digit)[0]  # Indices of all elements for this digit
        sampled_indices = rng.choice(digit_indices, size=n, replace=False)  # Randomly sampling n indices
        indices.extend(sampled_indices)  # Adding sampled indices to the list
    return Subset(dataset, indices)  # Returning subset of dataset with selected indices


train_dataset = reduce_mnist(full_train_dataset, 1000)
test_dataset = reduce_mnist(full_test_dataset, 200)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check data
for batch_idx, (images, labels) in enumerate(train_dataloader):
    print(f"Length of training data: {len(train_dataset)}")
    print(f"Batch {batch_idx}")
    print("Images shape:", images.size())  # Should be [batch_size, 1, 28, 28]
    print("Labels shape:", labels.size())
    plt.imshow(images[1].permute(1, 2, 0).numpy(), cmap='gray')
    plt.title(labels[1].item())
    break

# # Model Class
# ### Summary of Dimensions
#
# | Layer       | Output Shape          |
# |-------------|-----------------------|
# | Input       | (1, 28, 28)           |
# | Conv1       | (6, 24, 24)           |
# | Pool1       | (6, 12, 12)           |
# | Conv2       | (16, 8, 8)            |
# | Pool2       | (16, 4, 4)            |
# | Flatten     | (256)                 |
# | FC1         | (120)                 |
# | FC2         | (84)                  |
# | Output      | (10)                  |


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1: Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv1_out_dim = (self.conv1.out_channels,
                              (28 - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0]) // self.conv1.stride[0] + 1,
                              (28 - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0]) // self.conv1.stride[0] + 1)

        # Pooling Layer
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool1_out_dim = (self.conv1.out_channels,
                              (self.conv1_out_dim[1] - self.pool1.kernel_size + 2 * self.pool1.padding) // self.pool1.stride + 1,
                              (self.conv1_out_dim[1] - self.pool1.kernel_size + 2 * self.pool1.padding) // self.pool1.stride + 1)

        # Layer 2: Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=self.pool1_out_dim[0], out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv2_out_dim = (self.conv2.out_channels,
                              (self.pool1_out_dim[1] - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0]) // self.conv2.stride[0] + 1,
                              (self.pool1_out_dim[1] - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0]) // self.conv2.stride[0] + 1)

        # Pooling Layer
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2_out_dim = (self.conv2.out_channels,
                              (self.conv2_out_dim[1] - self.pool2.kernel_size + 2 * self.pool2.padding) // self.pool2.stride + 1,
                              (self.conv2_out_dim[1] - self.pool2.kernel_size + 2 * self.pool2.padding) // self.pool2.stride + 1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=int(np.prod(self.pool2_out_dim)), out_features=120)

        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=84)

        self.out_layer = nn.Linear(in_features=self.fc2.out_features, out_features=10)

    def forward(self, input_image):
        # Layer 1: Conv + ReLU + Pool
        layer1_out = self.pool1(F.relu(self.conv1(input_image)))

        # Layer 2: Conv + ReLU + Pool
        layer2_out = self.pool2(F.relu(self.conv2(layer1_out)))

        # Flatten the output for fully connected layers
        layer2_out = layer2_out.view(-1, int(np.prod(self.pool2_out_dim)))

        # Fully Connected Layers
        fc1_out = F.relu(self.fc1(layer2_out))
        fc2_out = F.relu(self.fc2(fc1_out))
        output = self.out_layer(fc2_out)  # Output: 10 classes

        return output


def check_accuracy(model, dataloader, data_length):
    # Function to test and calculate accuracy of a pre-trained model
    model.eval()
    correct = 0
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = F.softmax(model(inputs), dim=1)
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels).sum()
    return correct / data_length * 100


# Initialize the model, loss function, and optimizer
model = LeNet5()
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
# Training loop
num_epochs = 5  # Number of epochs to train
def train_loop(model, train_dataloader, num_epochs):
    train_time = time.time()  # Measure total time of training. This is only start time and will be subtracted from end.
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            # Zero the parameter gradients to avoid grads accumulation
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics (print loss every 100 batch)
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0
        print(f"Training Accuracy: {check_accuracy(model, train_dataloader, len(train_dataset)):.1f}%")
   
    return (time.time() - train_time)*1000  # Measure total time of training = end time - start time

train_time = train_loop(model, train_dataloader, num_epochs)
# Testing with 200 images per digit
test_time = time.time()  # Measure test time
test_accuracy = check_accuracy(model, test_dataloader, len(test_dataset))  # Testing and calculating accuracy
test_time = (time.time() - test_time)*1000

print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
print("Training complete!")
print(f"Training Accuracy: {check_accuracy(model, train_dataloader, len(train_dataset)):.1f}% | Testing Accuracy: {test_accuracy:.1f}%")
print(f"Training Time: {train_time:.1f} ms | Testing Time: {test_time:.1f} ms")
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
