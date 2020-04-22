#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DATASET
import torch
import pandas as pd
import torch.utils.data as Data
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

class dataset:

    CLASSES = 2
    TRAIN_SIZE = 576
    VALIDATION_SIZE = 123
    TEST_SIZE = 300
    TESTED_ELEMENTS = torch.tensor([250,50]).type(torch.FloatTensor) 
    TEST_RESULTS = torch.zeros(1,2)
    LAST_DATA_ELEMENT = 24
    file = ['train9.csv','validate9.csv','test9.csv']
    

    def CreateDataset(file, SIZE, LAST_DATA_COLUMN):
        file_reader = pd.read_csv(file, header=None)
        temp_tensor = torch.tensor(file_reader.values)

        #target = torch.zeros(SIZE, device = device)
        target = torch.zeros(SIZE)
        target = temp_tensor[:,LAST_DATA_COLUMN]
        #target = target.permute(1,0)
        target.requires_grad = False
        #target = torch.t(target).type(torch.LongTensor).cuda()
        target = torch.t(target).type(torch.LongTensor)

        #data = torch.zeros(1,1,SIZE,LAST_DATA_COLUMN, device = device)
        data = torch.zeros(1,1,SIZE,LAST_DATA_COLUMN)
        data[0,0,:,:] = temp_tensor[:,0:LAST_DATA_COLUMN]
        data = data.permute(2,3,1,0)
        data.requires_grad = False

        return data, target
    
    def CreateImageDataset(file, SIZE, LAST_DATA_COLUMN):
        file_reader = pd.read_csv(file, header=None)
        temp_tensor = torch.tensor(file_reader.values)
        Image = np.zeros((SIZE, LAST_DATA_COLUMN,LAST_DATA_COLUMN))
        
        target = torch.zeros(SIZE)
        target = temp_tensor[:,LAST_DATA_COLUMN]
        target.requires_grad = False
        target = torch.t(target).type(torch.LongTensor)
        # channel, size, h, w
        data2 = torch.zeros(1,SIZE,LAST_DATA_COLUMN,LAST_DATA_COLUMN) 
        data = torch.zeros(SIZE,LAST_DATA_COLUMN)
        data[:,:] = temp_tensor[:,0:LAST_DATA_COLUMN]
        #print(data)
        data = normalize(data, axis=1, norm='l1') * 100
        #print('normalized matrix: \n',data)
        for k in range (SIZE):
            for j in range (LAST_DATA_COLUMN):
                for i in range (LAST_DATA_COLUMN):
                    if ((data[k,j] / i) > 1) :
                        Image[k,i,j] = 1
        
        Image = torch.as_tensor(Image)
        for k in range (SIZE):        
            #print(Image)
            data2[0,k,:,:] = Image[k,:,:]
            #print(k,i,j)
            #plt.matshow(Image[k])
            #plt.ylabel('Credit Characteristics')
            #plt.colorbar()
            #plt.show()
            
        #data = torch.as_tensor(data)
        
        
        data2 = data2.permute(1,0,2,3)
        data2.requires_grad = False


        return data2, target

    train_data, train_target = CreateDataset(file[0], TRAIN_SIZE, LAST_DATA_ELEMENT)
    validation_data, validation_target = CreateDataset(file[1], VALIDATION_SIZE, LAST_DATA_ELEMENT)
    test_data, test_target = CreateDataset(file[2], TEST_SIZE, LAST_DATA_ELEMENT)

    train_dataset = Data.TensorDataset(train_data, train_target)
    validation_dataset = Data.TensorDataset(validation_data, validation_target)
    test_dataset = Data.TensorDataset(test_data, test_target)

    
    


#print('original Image: \n',Image)

#A = np.random.rand(1,N).astype(np.float64)
#print('original matrix: \n',A)




# In[14]:



            


# In[15]:



