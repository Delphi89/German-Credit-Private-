#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DATASET
import torch
import pandas as pd
import torch.utils.data as Data

class dataset:

    CLASSES = 2
    TRAIN_SIZE = 256
    VALIDATION_SIZE = 144
    TEST_SIZE = 599
    TESTED_ELEMENTS = torch.tensor([499,100]).type(torch.FloatTensor) 
    TEST_RESULTS = torch.zeros(1,2)
    LAST_DATA_ELEMENT = 24
    file = ['train6.csv','validate6.csv','test6.csv']

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

    train_data, train_target = CreateDataset(file[0], TRAIN_SIZE, LAST_DATA_ELEMENT)
    validation_data, validation_target = CreateDataset(file[1], VALIDATION_SIZE, LAST_DATA_ELEMENT)
    test_data, test_target = CreateDataset(file[2], TEST_SIZE, LAST_DATA_ELEMENT)

    train_dataset = Data.TensorDataset(train_data, train_target)
    validation_dataset = Data.TensorDataset(validation_data, validation_target)
    test_dataset = Data.TensorDataset(test_data, test_target)

