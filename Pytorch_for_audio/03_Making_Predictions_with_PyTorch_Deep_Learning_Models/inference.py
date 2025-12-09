import torch
from train import FeedForwardNet, download_mnist_dataset

class_mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected

if __name__ == "__main__":
    # load model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST
    _, validation_data = download_mnist_dataset()

    # get sample
    input, target = validation_data[0][0], validation_data[0][1]

    # make inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)

print(f"Pred: {predicted}, Expe: {expected}")
