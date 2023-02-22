import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


classes = [
    "Upper",
    "Lower",
    "Feet",
    "Bag"
]


def load_data():
    train_data = datasets.FashionMNIST(
        root='data',
        download=True,
        train=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )


    for X,y in train_data:
        print(y)
        break

    
    # merging labels by mapping previous to new labels
    label_mapping = {
        0 : 0,  # T-shirt/top -> Upper
        3 : 0,  # Dress -> Upper
        4 : 0,  # Coat  -> Upper
        6 : 0,  # Shirt -> Upper
        1 : 1,  # Trouser  -> Lower
        2 : 1,  # Pullover -> Lower
        5 : 2,  # Sandal     -> Feet
        7 : 2,  # Sneaker    -> Feet
        9 : 2,  # Ankle boot -> Feet
        8 : 3,  # Bag -> Bag
    }


    # new data by new labels
    new_train_data = [(X,label_mapping[i]) for X,i in train_data]
    new_test_data  = [(X,label_mapping[i]) for X,i in test_data]
    
    return new_train_data , new_test_data
    
    

def create_dataloaders(new_train_data, new_test_data):
    batchsize = 64
    train_dataloader = DataLoader(new_train_data,batch_size=batchsize)
    test_dataloader = DataLoader(new_test_data,batch_size=batchsize)


    for X, y in train_dataloader:
        print(y)
        break

    for X, y in test_dataloader:
        print(y)
        break

    for X,y in test_dataloader:
        print(f"X shape {X.shape}")
        print(f"y shape {y.shape}")
        break
     
    return train_dataloader, test_dataloader


# model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 20),
                nn.ReLU(),
                nn.Linear(20,4)
            )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_model():
    model = NeuralNetwork().to(device)
    return model

def get_lossfn_and_optimizer(mymodel):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=1e-3)
    return loss_fn , optimizer


def traind(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def testd(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:-\n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(train_dataloader, test_dataloader, model1, loss_fn1, optimizer1, epochs=5):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        traind(train_dataloader, model1, loss_fn1, optimizer1)
        testd(test_dataloader, model1, loss_fn1)
    print("Done!")
    return model1


def save_model(model1,mypath="model.pth"):
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    

def load_model(mypath="model.pth"):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    return model



def sample_test1(model1, new_test_data):
    model1.eval()
    x, y = new_test_data[0][0], new_test_data[0][1]
    with torch.no_grad():
        pred = model1(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

def sample_test2(model1, new_test_data):  
    model1.eval()
    x, y = new_test_data[0][0], new_test_data[0][1]
    gy=0
    for i in range(4) :
        x, y = new_test_data[gy][0], new_test_data[gy][1]
        with torch.no_grad():
            pred = model1(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"\n')
    gy=gy+10


#cs20b035
