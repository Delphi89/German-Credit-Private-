#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, patience=40, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            #print('Resetting Patience Counter')

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Average Validation Loss Decreased ({self.val_loss_min:.2f} --> {val_loss:.2f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# In[2]:


class ReducedRate:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, waiting=10, verbose=False):
        self.waiting = waiting
        self.verbose = verbose
        self.counter2 = 0
        self.best_score2 = None
        self.early_stop2 = False

    def __call__(self, val_loss, model):

        score2 = -val_loss

        if self.best_score2 is None:
            self.best_score2 = score2
        elif score2 < self.best_score2:
            self.counter2 += 1
            print(f'ReducedRate counter: {self.counter2} out of {self.waiting}')
            if self.counter2 >= self.waiting:
                self.early_stop2 = True
                self.counter2 = 0
            else:
                self.early_stop2 = False    
        else:
            self.best_score2 = score2
            self.counter2 = 0
            self.early_stop2 = False
            #print('Resetting Patience Counter')


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#print(torch.cuda.nccl.is_available())
torch.manual_seed(0)   # reproducible


# In[4]:


patience = 100              # if validation loss not going down, wait "patience" number of epochs
waiting = 51
train_losses = []         # to track the training loss as the model trains
early_stopping = EarlyStopping(patience=patience, verbose=True)  # initialize the early_stopping object
reduced_rate = ReducedRate(waiting=waiting, verbose=True)

# Hyper Parameters
EPOCHS = 3000              # number of epochs
BATCH_SIZE = 256        # used for training 
LR  = 8e-3                 # learning rate
LR2 = 8e-3
MM = 0.6                  # momentum - used only with SGD optimizer

DROPOUT_INIT = 0
DROPOUT_MIDDLE_1 = DROPOUT_INIT
DROPOUT_MIDDLE_2 = DROPOUT_INIT
DROPOUT_CLASSIFIER = DROPOUT_INIT
DROPOUT_SKIP = DROPOUT_INIT
REGULARIZATION = 0.003

RELU = True

L_FIRST = 24
L_SECOND = 24
L_FIRST2 = L_FIRST
L_SECOND2 = L_SECOND
L_THIRD = L_SECOND
L_FIFTH = L_THIRD 
L_SIXTH = L_THIRD
L_SEVENTH = L_SECOND 
L_SEVENTH2 = L_SECOND 

TRAIN_VECTOR_SIZE = 255
TRAIN_SIZE = 255
VALIDATION_SIZE = 89
TEST_SIZE = 653
min_val_loss = 999999

accuracy = 0
label1 = torch.tensor([527,127]).type(torch.FloatTensor) 
label2 = torch.zeros(1,2)

# load training set
train = pd.read_csv('train5.csv')
train_tensor = torch.tensor(train.values)

train_target = torch.zeros(1,255)
train_target[0,:] = train_tensor[:,24]
train_target = train_target.permute(1,0)
train_target.requires_grad = False
train_target = torch.t(train_target).type(torch.LongTensor).cuda()

train_data = torch.zeros(1,1,255,24)
train_data[0,0,:,:] = train_tensor[:,0:24]
train_data = train_data.permute(2,3,1,0)
train_data.requires_grad = False


# load validation set
validation = pd.read_csv('validate5.csv')
validation_tensor = torch.tensor(validation.values)

validation_target = torch.zeros(1,89)
validation_target[0,:] = validation_tensor[:,24]
validation_target = validation_target.permute(1,0)
validation_target.requires_grad = False
validation_target = torch.t(validation_target).type(torch.LongTensor).cuda()

validation_data = torch.zeros(1,1,89,24)
validation_data[0,0,:,:] = validation_tensor[:,0:24]
validation_data = validation_data.permute(2,3,1,0)
validation_data.requires_grad = False


# load test set
test = pd.read_csv('test5.csv')
test_tensor = torch.tensor(test.values)

test_target = torch.zeros(1,653)
test_target[0,:] = test_tensor[:,24]
test_target = test_target.permute(1,0)
test_target.requires_grad = False
test_target = torch.t(test_target).type(torch.LongTensor).cuda()

test_data = torch.zeros(1,1,653,24)
test_data[0,0,:,:] = test_tensor[:,0:24]
test_data = test_data.permute(2,3,1,0)
test_data.requires_grad = False
#test_data = test_data.permute(1,0)
#print("test data...", test_data)



# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = False, 
                               num_workers = 0, pin_memory=True)
valid_loader = Data.DataLoader(dataset = validation_data, batch_size = BATCH_SIZE, shuffle = False, 
                               num_workers = 0, pin_memory=True)


# In[5]:



# Counter for the execution time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
         
        self.linear_1 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, L_SECOND,1, bias=False),
            nn.ReLU(RELU), 
            nn.Dropout(p = DROPOUT_INIT),
        )
            
        self.linear_2 = nn.Sequential(      
            nn.Linear(L_SECOND, L_SECOND, bias=True),
            nn.ReLU(RELU), 
            nn.Dropout(p = DROPOUT_INIT),
            nn.Linear(L_SECOND, L_SECOND, bias=True),
            nn.ReLU(RELU), 
            nn.Dropout(p = DROPOUT_INIT),
            nn.Linear(L_SECOND, L_SECOND, bias=True),
            nn.ReLU(RELU), 
            nn.Dropout(p = DROPOUT_INIT),
            nn.Linear(L_SECOND, 2, bias=True),
            nn.LogSoftmax(1)
        )  
        

    def forward(self, x): 
        x1 = self.linear_1(x)
        x2 = x1.view(x1.size(0), -1)
        output = self.linear_2(x2)
        #x1 = self.conv_init(x)  
        #x2 = self.conv2(x1)   + self.conv3(x1)   + self.conv4(x1)
        #x3 = self.conv2L2(x1) + self.conv3L2(x1) + self.conv4L2(x1)
        #x4 = self.conv2L2(x1) + self.conv3L2(x1) + self.conv4L2(x1)
        #x5 = x1 + x2 + x3 + x4
        #x6 =  self.conv2L2(x5) + self.conv3L2(x5) + self.conv4L2(x5)
        #x7 =  self.conv2L2(x5) + self.conv3L2(x5) + self.conv4L2(x5)
        #x8 = x1 + x6 + x7
        #x9 = x8.view(x8.size(0), -1) 
        #output = self.classifier(x9)
        return output, x
      

cnn = CNN()
print(cnn)  # net architecture
list(cnn.parameters())

loss_func2 = nn.CrossEntropyLoss().cuda()
#loss_func1 = nn.NLLLoss().cuda()
loss_func1 = nn.MultiMarginLoss().cuda()


#optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION) # 
#optimizer = torch.optim.Adadelta(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION)
#optimizer = torch.optim.Adagrad(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION) 
optimizer = optim.SGD(cnn.parameters(), lr=LR, momentum=MM, weight_decay=REGULARIZATION)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "NVIDIA 1080TI GPUs!")
    cnn = nn.DataParallel(cnn)

cnn = torch.nn.DataParallel(cnn, device_ids=[0]).cuda()
cnn.to(device)

train_data, train_target = train_data.to(device), train_target.to(device)
validation_data, validation_target = validation_data.to(device), validation_target.to(device)
test_data, test_target = test_data.to(device), test_target.to(device)


def train():
    cnn.train()
    for step, data in enumerate(train_loader,0):
        optimizer.zero_grad()                         # clear the gradients of all optimized variables            
        output, temp = cnn(train_data)                # forward pass: compute predicted outputs by passing inputs to the model               
        loss1 = loss_func1(output, train_target[0])        # calculate the loss
        loss2 = loss_func2(output, train_target[0])
        loss = loss1 + loss2
        train_losses.append(loss.item())              # record training loss                           
        loss.backward()                               # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()                              # perform a single optimization step (parameter update)
    return output, loss
        
def validation(cnn, valid_loader, valid_loss_func1, valid_loss_func2):
    valid_loss = 0
    accuracy = 0

    for inputs, classes in enumerate(valid_loader,0):
        output, temp = cnn(validation_data)            # forward pass: compute predicted outputs by passing inputs to the model
        valid_loss += valid_loss_func1(output, validation_target[0]).item() + valid_loss_func2(output, validation_target[0]).item()
        ps = torch.exp(output)
        equality = (validation_target[0].data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy           


test_loss = min_val_loss;
# training 
for epoch in range(EPOCHS):
    loss = None
        
    train_output, train_loss = train()   
    cnn.eval()                 # switch to evaluation (no change) mode
    with torch.no_grad():
        valid_loss, valid_accuracy = validation(cnn, valid_loader, loss_func1, loss_func2)

        
    test_output, last_layer = cnn(train_data)
    pred_y = torch.max(test_output, 1)[1]
    print("---------------------")
    print("Epoch: {}/{}.. ".format(epoch+1, EPOCHS))
    print("Training Accuracy  : {:.2f} ".format(torch.sum(pred_y == train_target[0]).type(torch.FloatTensor) / float(train_target.size(1))),
        "Validation Accuracy: {:.2f}".format(valid_accuracy/len(valid_loader)))
    
    #epoch_len = len(str(EPOCHS))  
   
    print_msg = (f' Training Loss  : {train_loss:.2f} ' +
                 f'  Validation Loss: {valid_loss:.2f}')
        
    print(print_msg)
       
    # clear lists to track next epoch
    train_losses = []
        
    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, cnn)
    reduced_rate(valid_loss, cnn)
    
    if reduced_rate.early_stop2:
        print("Reducing Learning Rate")
        LR  = LR  / 10
        LR2 = LR2 / 10
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
print("Loaded the model with the lowest Validation Loss!")
cnn.load_state_dict(torch.load('checkpoint.pt'))        
    
    
#test           
with torch.no_grad():
    outputs, temp = cnn(test_data)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.to(device)

    c = (predicted == test_target[0]).squeeze()
    c = c.to(device)

    for j in range (TEST_SIZE):
        if (c[j] == 1): 
            label2[0,predicted[j]] += 1 
            
#print('Correct classif. in each class  : ',label2)
#print('Total number of pixels per class: ',label1)            
percent = (torch.sum(c).item()/TEST_SIZE)
print('Correct Classification Percent: ',percent)
print('Results by class: ',label2/label1)

end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print('Total execution time (minutes): ',start.elapsed_time(end)/60000)

torch.cuda.empty_cache()


# In[ ]:





# In[ ]:




