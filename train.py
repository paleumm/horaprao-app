import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torchsampler import ImbalancedDatasetSampler
from ResNet import ResNet50

train_path = "./input/train"
test_path = "./input/val"
batch_size = 256
lr = 1e-3

train_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomApply([
                    transforms.RandomRotation(180)                    
                ]),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
])

transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=batch_size,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    sampler=ImbalancedDatasetSampler(test_dataset),
    batch_size=batch_size,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = 'cpu'
print(f'using device : {device}')

model = ResNet50(num_classes=2).to(device)
# print(model)

num_data = 643 + 424
w_horapa = 643/num_data
w_kapao = 424/num_data
class_weight = [w_horapa, w_kapao]
class_weight = torch.FloatTensor(class_weight)

# criterion
# criterion = nn.BCEWithLogitsLoss(weight=class_weight)
criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
gamma = 0.7
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 5 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
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
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------")
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")
