import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training = True):
    custom_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data',train=False, transform=custom_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)

    if (training):
        return train_loader
    else:
        return test_loader


def build_model():
    model = nn.Sequential (
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )
    return model


def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()

        for i, data in enumerate(train_loader, 0):
            images, labels = data
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        # print statistics
        print(f'Train Epoch: {epoch}   Accuracy: {correct}/{total} ({100*(correct/total):.2f})   Loss: {(running_loss / total)*train_loader.batch_size:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss = criterion(outputs, labels)
            running_loss += running_loss.item()


    if (show_loss):
        print(f'Average loss: {running_loss:.4f}')
    
    print(f'Accuracy: {100 * correct/len(test_loader.dataset):.2f}%')
    

def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    
    image = model(test_images)
    prob = F.softmax(image, dim=1)
    max = 0

    for i in range(3):
        for x in range(len(prob[index])):
            if prob[index][x] > prob[index][max]:
                max = x
        print(f'{class_names[max]}: {100*prob[index][max]:.2f}%')
        prob[index][max] = -1
        max = 0

def main():
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)

    model = build_model()

    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion)
    
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)


if __name__=="__main__":
    main()