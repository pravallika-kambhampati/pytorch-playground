"""
Decoupling dataset code from our model training code 
PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset 
that allow you to use pre-loaded datasets as well as your own data. 
"""

"""
Dataset stores the samples and their corresponding labels, 
and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Iterating and visualizing the datasets

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# Preparing your data for training with DataLoaders

"""
The Dataset retrieves our dataset’s features and labels one sample at a time. 
While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, 
and use Python’s multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.
"""

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
# Because we specified shuffle=True, after we iterate over all batches the data is shuffled (for finer-grained control over the data loading order, take a look at Samplers).

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Extracting first image from the batch and removes any singleton dimensions,
# resulting in a 2D array. This is a common step when working with image data, 
# as it removes extra dimensions that are not necessary for displaying the image.
img = train_features[0].squeeze()

# extracts the first label from the batch of labels
label = train_labels[0]

# displaying the first image extracted in grayscale
plt.imshow(img, cmap="gray")
plt.show()

# printing the label of the image
print(f"Label: {label}")