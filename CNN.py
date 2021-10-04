import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 5 * 5, 480)
        self.fc2 = nn.Linear(480, 84)
        self.fc3 = nn.Linear(84, 10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return self.logsoftmax(out)


def main():
    save_frequency = 1
    num_epochs = 20
    num_print = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.4)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('data/', train=True, transform=trans, download=True)
    test_dataset = datasets.CIFAR10('data/', train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    for epoch in range(num_epochs):
        for t, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(device), Variable(target).to(device)
            pred = model(data)
            loss = loss_fn(pred, target)
            if (t + 1) % num_print == 0:
                print('{}/{} Train loss: {:.4f} '.format(epoch+1, t+1, loss.data.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % save_frequency == 0:
            torch.save(model.state_dict(), 'CNN.pt'.format(epoch))

        correct = 0
        for data, target in test_loader:
            data, target = Variable(data, volatile=True).to(device), Variable(target).to(device)
            output = model(data)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('Epoch:{} Test accuracy: {:.4f}% \n'.format(epoch+1, 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
