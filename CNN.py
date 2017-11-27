from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import cv2
import shutil
from PIL import Image


# In[3]:

'''
Get train data
'''
# training_data = pickle.load(open('training_data.pkl','r'))
training_data = pickle.load(open('train_data_new.pkl','r'))

training_data = np.array(training_data)
np.random.shuffle(training_data)


# In[4]:

'''
Get Validation data
'''
# validation_data = pickle.load(open('validation_data.pkl','r'))
validation_data = pickle.load(open('validation_data_new.pkl','r'))

validation_data = np.array(validation_data)


# In[5]:

'''
Get Test data
'''
# test_data = pickle.load(open('test_data.pkl','r'))
test_data = pickle.load(open('test_data_new.pkl','r'))

test_data = np.array(test_data)


# In[6]:

class Net(nn.Module):

    def __init__(self, INPUT_H_DIM, INPUT_W_DIM): 
        super(Net, self).__init__()
        CONV_1_NUM = 4
        CONV_1_DIM = 3
        stride = 1
        FC_1_DIM = 100
        # 1 input image channel 10x10, 4 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, CONV_1_NUM, CONV_1_DIM)
#         self.dropout = nn.Dropout(0.2)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(4 * 8 * 8, 100)
        CONV1_H_DIM = ((INPUT_H_DIM - CONV_1_DIM)/stride) + 1
        CONV1_W_DIM = ((INPUT_W_DIM - CONV_1_DIM)/stride) + 1
        self.fc1 = nn.Linear(CONV_1_NUM * CONV1_H_DIM * CONV1_W_DIM, FC_1_DIM)
        self.fc2 = nn.Linear(FC_1_DIM, 1) #Number of classes = 'text'
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
#         x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.sigmoid(x)
#         out = F.log_softmax(x)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[7]:

# Preparing the data
trainloader = DataLoader(training_data.tolist(), batch_size=1, shuffle=True)
validloader = DataLoader(validation_data.tolist())
testloader = DataLoader(test_data.tolist())


# In[10]:

def validation_function(validloader, net, optimizer, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    net.eval()
    for i, data in enumerate(validloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels_var = Variable(inputs.unsqueeze(0).float(), volatile=True), Variable(labels.float(), volatile=True)

        outputs = net(inputs)
        loss = criterion(outputs, labels_var.unsqueeze(-1))

        # print statistics
        running_loss += loss.data[0]
        
        predicted = torch.ge(outputs.data, torch.FloatTensor([0.5])) # Because batch size is 1
        total += labels.size(0)
        correct += (predicted.long() == labels).sum()
    
    print_loss = running_loss/float(i+1)
    print_acc = 100 * correct / float(total)
    print("*****validation*** running_loss = ", print_loss)
    print('Accuracy of the network on the 1728 validation images: %d %%' % (print_acc))
    
    return print_loss, print_acc
    


# In[11]:

def test_function(testloader, net, optimizer, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    net.eval()
    predicted_list = []
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
#         inputs, labels_var = Variable(inputs.unsqueeze(0).float()), Variable(labels.long())
        inputs, labels_var = Variable(inputs.unsqueeze(0).float()), Variable(labels.float())

        outputs = net(inputs)
#         loss = criterion(outputs, labels_var)
        loss = criterion(outputs, labels_var.unsqueeze(-1))
        
        # print statistics
        running_loss += loss.data[0]

        predicted = torch.ge(outputs.data, torch.FloatTensor([0.5])) # Because batch size is 1
        predicted_list.append(predicted)
        total += labels.size(0)
        correct += (predicted.long() == labels).sum()
    
    print("******test*** running_loss = ",running_loss/float(i+1))
    print('Accuracy of the network on the 10368 test images: %d %%' % (100 * correct / float(total)))
    return predicted_list
    


# In[12]:

def train_function(trainloader, net, optimizer, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    net.train()
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        # wrap them in Variable
        inputs, labels_var = Variable(inputs.unsqueeze(0).float()), Variable(labels.float())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels_var.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data.numpy()[0]
    
        predicted = torch.ge(outputs.data, torch.FloatTensor([0.5])) # Because batch size is 1
        total += labels.size(0)
        correct += (predicted.long() == labels).sum()
        
    print_loss = running_loss/float(i+1)
    print_acc = 100 * correct / float(total)
    print("******train*** running_loss = ", print_loss)
    print('Accuracy of the network on train images: %d %%' % (print_acc))
    
    return print_loss, print_acc 


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Declaration
# 1 input image channel 14x14, 4 output channels, 3x3 square convolution
INPUT_IMAGE_DIM = training_data[0][0].shape
INPUT_H_DIM = INPUT_IMAGE_DIM[0]
INPUT_W_DIM = INPUT_IMAGE_DIM[1]


# Training Phase
net = Net(INPUT_H_DIM, INPUT_W_DIM)
# print(net)
# criterion =  nn.NLLLoss() #nn.CrossEntropyLoss()
criterion =  nn.BCELoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.001)
optimizer.zero_grad()

train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []
NUM_EPOCHS = 25
best_valid_acc = 0

for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    train_loss, train_acc = train_function(trainloader, net, optimizer, criterion)
    valid_loss, valid_acc = validation_function(validloader, net, optimizer, criterion)
    # Save model if it has better accuracy
    is_best = valid_acc > best_valid_acc
    best_valid_acc = max(valid_acc, best_valid_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'best_prec1': best_valid_acc,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    
#     predicted_list = test_function(testloader, net, optimizer, criterion)
    print("------------------------------------")

plt.plot(np.arange(1,NUM_EPOCHS + 1), train_acc_list)
plt.plot(np.arange(1,NUM_EPOCHS + 1), valid_acc_list)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='lower right')
plt.show()

plt.plot(np.arange(1,NUM_EPOCHS + 1), train_loss_list)
plt.plot(np.arange(1,NUM_EPOCHS + 1), valid_loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
plt.show()


if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))