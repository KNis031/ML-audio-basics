import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCHSIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
                nn.Linear(28*28, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
                )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)

        return predictions


# download dataset 
def download_mnist_dataset():
    train_data = datasets.MNIST(
            root = "data",
            download = True,
            train = True,
            transform = ToTensor()
            )

    validation_data= datasets.MNIST(
            root = "data",
            download = True,
            train = False,
            transform = ToTensor()
            )

    return train_data, validation_data

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range (epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("----------------------")
    print("Training done!")


if __name__ == '__main__':
    train_data, _ = download_mnist_dataset()
    print("MNIST DATA DOWNLOADED")

    train_data_loader = DataLoader(train_data, batch_size=BATCHSIZE)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    feed_forward_net = FeedForwardNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, epochs=EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("model saved!")
