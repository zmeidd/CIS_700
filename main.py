from pytools import average
from dataloader import *
import torch
from torch.utils.data import Dataset, DataLoader
from Network import *
import numpy as np
import matplotlib.pyplot as plt
batch_size = 64
epochs = 10

train_acc = []
test_acc = []
train_loss = []
test_loss = []
val_acc = []
val_loss = []
norm = []

def train(model, device, dataloader, optimizer, criterion, epoch):
    model.train()
    sample = 0
    correct_num = 0
    total_loss = []
    for i_batch, sample_batched in enumerate(dataloader):
        pivot = sample_batched[0].to(device)
        statement = sample_batched[1].to(device)
        label = sample_batched[2].to(device)
        sample += pivot.shape[0]
        optimizer.zero_grad()
        output = model(pivot, statement)
        loss = criterion(output, label)
        total_loss.append(loss)
        loss.backward()
        optimizer.step()
        prediction = (output > 0.5).int()
        difference = label - prediction
        correct_num += torch.sum(difference == 0)
        if i_batch % 200 == 0:
            acc = correct_num/sample
            print('Train epoch: {} [{}/{}] \tLoss: {} \tAccuracy: {}'.format(epoch, i_batch,len(dataloader), loss, acc))
    train_acc.append(correct_num/len(dataloader.dataset))
    train_loss.append(min(total_loss).detach().cpu().numpy())



def test(model, device, dataloader, optimizer, criterion, epoch):
    model.eval()
    total_loss = 0
    samples = 0
    correct_num = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            pivot = sample_batched[0].to(device)
            statement = sample_batched[1].to(device)
            samples += pivot.shape[0]
            label = sample_batched[2].to(device)
            output = model(pivot, statement)
            loss = criterion(output, label)
            total_loss += loss
            prediction = (output > 0.5).int()
            difference = label - prediction
            correct_num += torch.sum(difference == 0)
        loss = total_loss / samples
        acc = correct_num/samples
        print('Average Test Loss: {} \tAverage Test Accuracy: {}'.format(loss, acc))
        test_loss.append(loss.detach().cpu().numpy())
        test_acc.append(acc.detach().cpu().numpy())


def validation(model, device, dataloader, optimizer, criterion, epoch):
    model.eval()
    total_loss = 0
    samples = 0
    correct_num = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            pivot = sample_batched[0].to(device)
            statement = sample_batched[1].to(device)
            samples += pivot.shape[0]
            label = sample_batched[2].to(device)
            output = model(pivot, statement)
            loss = criterion(output, label)
            total_loss += loss
            prediction = (output > 0.5).int()
            difference = label - prediction
            correct_num += torch.sum(difference == 0)
        loss = total_loss / samples
        acc = correct_num / samples
        print('Average Validation Loss: {} \tAverage Test Accuracy: {}'.format(loss, acc))
        val_acc.append(acc.detach().cpu().numpy())
        val_loss.append(loss.detach().cpu().numpy())
        return acc



def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_path = 'train_data.npz'
    validation_path = 'validation_data.npz'
    validation_dataset = C2xDataset(validation_path)
    train_test_dataset = C2xDataset(train_path)
    train_size = int(len(train_test_dataset) / 6 * 5)
    test_size = len(train_test_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_test_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = CNN_2X().to(device)
    # optimzier = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    optimzier = torch.optim.SGD(model.parameters(),lr= 0.01)
    # optimzier = torch.optim.Adam(model.parameters(),lr= 0.0001)
    criterion = torch.nn.BCELoss()
    best_loss = np.inf
    total_norm = 0
    for epoch in range(epochs):
        train(model, device, train_loader, optimzier, criterion, epoch)
        test(model, device, test_loader, optimzier, criterion, epoch)
        loss = validation(model, device, validation_loader, optimzier, criterion, epoch)
        if loss < best_loss:
            best_loss = loss
        print("Best Loss: {}".format(best_loss))
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        norm.append(total_norm)



if __name__ == '__main__':
    main()
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    fig.suptitle('Accuracy', fontsize=20)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('acc', fontsize=18)
    t_acc = []
    for i in range(len(train_acc)):
        t_acc.append(train_acc[i].detach().cpu().numpy())
    plt.plot(np.arange(1,len(train_acc)+1), t_acc, label = "train acc")
    plt.plot(np.arange(1,len(train_acc)+1), test_acc, label = "test acc")
    plt.plot(np.arange(1,len(train_acc)+1), val_acc, label = "validation acc")
    plt.legend()
   

    plt.subplot(1, 3, 2)
    fig.suptitle('Loss', fontsize=20)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.plot(np.arange(1,len(train_acc)+1), train_loss, label = "train loss")
    plt.plot(np.arange(1, len(test_acc)+1), test_loss, label = "test loss")
    plt.plot(np.arange(1,len(val_acc)+1), val_loss, label = "validation loss")
    plt.legend()


    plt.subplot(1, 3, 3)
    fig.suptitle('Norm gradient', fontsize=20)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('gradient', fontsize=18)
    plt.plot(np.arange(1,len(train_acc)+1), norm, label = "norm gradients")
    plt.legend()
    plt.show()