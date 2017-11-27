
# coding: utf-8

# In[2]:

# The architecture of our CNN is given in Figure 1. The structure
# can be summarized as 10×10x1−26×26×4−100−M,
# where M is the number of classes. The input is a grayscale
# image patch. The size of the image patch is 28×28 pixels. Our
# CNN architecture contains only one convolution layer which
# consists of 4 kernels. The size of each kernel is 3 × 3 pixels.
# Unlike other traditional CNN architecture, the pooling layer is
# not used in our architecture. Then one fully connected layer
# of 100 neurons follows the convolution layer. The last layer
# consists of a logistic regression with softmax which outputs
# the probability of each class, such that


# In[218]:

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
from PIL import Image


# In[231]:

'''
Get train data
'''
training_data = pickle.load(open('training_data.pkl','r'))

training_data = np.array(training_data)
training_data = training_data[:20] #Test on first 2000 image segments

X_Train = training_data[:,0]
y_Train = training_data[:,1]

training_data.shape


# In[232]:

'''
Get Validation data
'''
validation_data = pickle.load(open('validation_data.pkl','r'))

validation_data = np.array(validation_data)
validation_data = validation_data 

X_Valid = validation_data[:,0]
y_Valid = validation_data[:,1]

validation_data.shape


# In[233]:

'''
Get Test data
'''
test_data = pickle.load(open('test_data.pkl','r'))

test_data = np.array(test_data)
# validation_data = validation_data 

X_Test = test_data[:,0]
y_Test = test_data[:,1]

test_data.shape


# In[259]:

class Net(nn.Module):

    def __init__(self): # DO NOT HARDCODE
        super(Net, self).__init__()
        # 1 input image channel 10x10, 4 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 4, 3)
        #self.dropout = nn.Dropout(0.5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 2) #Number of classes = 'text'
#         self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
        #         dropout with 0.5
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #out = self.sigmoid(x)
        x = F.log_softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[260]:

net = Net()
print(net)


# In[261]:

# Total number of learnable parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())


# In[262]:

# Preparing the data
trainloader = DataLoader(training_data.tolist(), batch_size=1, shuffle=True)
validloader = DataLoader(validation_data.tolist())
testloader = DataLoader(test_data.tolist())


# In[263]:

# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()


# In[ ]:


    


# In[264]:

def validation_function(validloader, net, optimizer, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    net.eval()
    for i, data in enumerate(validloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels_var = Variable(inputs.unsqueeze(0).float(), volatile=True), Variable(labels.long(), volatile=True)

        outputs = net(inputs)
        loss = criterion(outputs, labels_var)

        # print statistics
        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("****** running_loss = ",running_loss)
    print('Accuracy of the network on the 1728 validation images: %d %%' % (100 * correct / total))
    


# In[265]:

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
        inputs, labels_var = Variable(inputs.unsqueeze(0).float()), Variable(labels.long())

        outputs = net(inputs)
        loss = criterion(outputs, labels_var)
        
        # print statistics
        running_loss += loss.data[0]

        _, predicted = torch.max(outputs.data, 1)
        predicted_list.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("****** running_loss = ",running_loss)
    print('Accuracy of the network on the 10368 test images: %d %%' % (100 * correct / total))
    return predicted_list
    


# In[266]:

def train_function(trainloader, net, optimizer, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    net.train()
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        # wrap them in Variable
        inputs, labels_var = Variable(inputs.unsqueeze(0).float()), Variable(labels.long())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels_var)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
        _, predicted = torch.max(outputs.data, 1)
#         print ('predicted: %d' % (predicted))
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("****** running_loss = ",running_loss)
    print('Accuracy of the network on the 21643 train images: %d %%' % (100 * correct / total))
    return 


# In[267]:

# Training Phase
# net = Net()
for epoch in range(5):  # loop over the dataset multiple times
    train_function(trainloader, net, optimizer, criterion)
    # validation_function(validloader, net, optimizer, criterion)
    # predicted_list = test_function(testloader, net, optimizer, criterion)
    print ("---------------------------")


# In[ ]:




# In[144]:

# IGNORE THE CODE FROM HERE

