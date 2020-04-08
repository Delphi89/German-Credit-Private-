#!/usr/bin/env python
# coding: utf-8

# best score: 278

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
import numpy as np
import os
import cProfile, pstats, io
from pstats import SortKey
from random import randint
from Utillities5 import Utillities
from cnn_model6 import CNN6     
import matplotlib.pyplot as mp
#import matplotlib.pyplot as plt


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if os.path.exists("checkpoint.pt"):
    os.remove("checkpoint.pt")

torch.manual_seed(70)   # reproducible

OPTIMIZATION_PLUGIN = 'Bayesian' # 'Bayesian' or 'Scikit' or 'GradDescent'
#Bayesian requires: $ conda install -c conda-forge bayesian-optimization

GET_STATS = False
GPU_SELECT = 2 # can be 0, 1, 2 (both)
PARALLEL_PROCESSES = 2
TRIALS = 1
RANDOM_STARTS = 5000
LR  = 1e-5                    # learning rate
SCI_LR =  1e-5
LR2 = 1e-5
SCI_MM = 0.5                  # momentum - used only with SGD optimizer
MM = 0.5
L_FIRST = 24                  # initial number of channels
KERNEL_X = 24
patience = 5                 # if validation loss not going down, wait "patience" number of epochs
accuracy = 0
MaxCredit = -800

CreditVector = np.zeros(RANDOM_STARTS + TRIALS)
CreditVector = CreditVector - 800
CreditVec = np.zeros(RANDOM_STARTS + TRIALS)
count = 0

pr = cProfile.Profile()

if GET_STATS:
    pr.enable()
    

if GPU_SELECT == 2:
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using", torch.cuda.device_count(), "NVIDIA 1080TI GPUs!")

if GPU_SELECT == 1:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")    
    print("Using one (the second) NVIDIA 1080TI GPU!")

if GPU_SELECT == 0:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       
    print("Using one (the first) NVIDIA 1080TI GPU!")


# In[2]:


from early_stopping import EarlyStopping
from dataset2 import dataset


early_stopping = EarlyStopping(patience=patience, verbose=True)  # initialize the early_stopping object

# Counter for the execution time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()





if OPTIMIZATION_PLUGIN == 'Bayesian' :
    from bayes_opt import BayesianOptimization
    
    #def black_box_function(x, y):
    def objective(SCI_RELU, SCI_BIAS, SCI_loss_type,
                  SCI_optimizer, SCI_LR, SCI_MM, 
                  SCI_REGULARIZATION, SCI_EPOCHS, SCI_BATCH_SIZE, 
                  SCI_DROPOUT, SCI_L_SECOND, SCI_BN_MOMENTUM, SCI_SGD_MOMENTUM,
                  SCI_BN_EPS, SCI_BN_STATS, SCI_LAST_LAYER, SCI_ACT_LAYER):
        global count, PercentVector, PercentVec, device, MaxCredit

        
        SCI_BATCH_SIZE = int(SCI_BATCH_SIZE)   
        SCI_LAST_LAYER = int(SCI_LAST_LAYER)       
        SCI_ACT_LAYER =int(SCI_ACT_LAYER)
        SCI_MM = round(SCI_MM,3)                                # real with three decimals between (0.001, 0.999)
        SCI_LR = round(SCI_LR,5)                                # real with five decimals between(1e-4, 7e-1)            
        SCI_DROPOUT = round(SCI_DROPOUT,2)                      # real with two decimals between (0, 0.4)
        SCI_L_SECOND = int(SCI_L_SECOND)                        # integer between 2 and 64
        SCI_EPOCHS = int(SCI_EPOCHS)                            # integer between (100, 500)
        SCI_BN_MOMENTUM = round(SCI_BN_MOMENTUM,2)              # real with two decimals between (0, 0.99)
        SCI_SGD_MOMENTUM = round(SCI_SGD_MOMENTUM,2)            # real with two decimals between (0, 0.99) 
        SCI_loss_type = int(SCI_loss_type)                      # integer between 1 and 3 ('CrossEntropyLoss', 'MultiMarginLoss','NLLLoss')
        SCI_BN_EPS = int(SCI_BN_EPS)

            
        if int(SCI_RELU) == 1 :                                 # integer between 1 and 2 ('True', 'False')
            SCI_RELU = True      
        else:
            SCI_RELU = False      
        if int(SCI_BIAS) == 1 :                                 # integer between 1 and 2 ('True', 'False')
            SCI_BIAS = True      
        else:
            SCI_BIAS = False  
 
        SCI_REGULARIZATION = float(str(SCI_REGULARIZATION))
    
    
  
        if SCI_BN_EPS == 0:
            BN_EPS = 5e-6
        if SCI_BN_EPS == 1:
            BN_EPS = 1e-5
        if SCI_BN_EPS == 2:
            BN_EPS = 5e-6
        if SCI_BN_EPS == 3:
            BN_EPS = 1e-6             
        if SCI_BN_EPS == 4:
            BN_EPS = 5e-7     
        if SCI_BN_EPS == 5:
            BN_EPS = 1e-7
        if SCI_BN_EPS == 6:
            BN_EPS = 5e-8
        if SCI_BN_EPS == 7:
            BN_EPS = 1e-8
        if SCI_BN_EPS == 8:
            BN_EPS = 3e-7             
        if SCI_BN_EPS == 9:
            BN_EPS = 8e-7                     
        print('BN Batch EPS: ', BN_EPS)    
        
        SCI_BN_STATS = int(SCI_BN_STATS)
        if SCI_BN_STATS == 0:
            BN_STATS = True
        if SCI_BN_STATS == 1:
            BN_STATS = False
        print('BN Batch STATS: ', BN_STATS)     
            
        
        
        cnn = CNN6(L_FIRST, SCI_L_SECOND, KERNEL_X,
                   SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU,
                   SCI_DROPOUT, dataset.CLASSES,
                   BN_EPS, BN_STATS, SCI_LAST_LAYER, SCI_ACT_LAYER)     

        optimizer = Utillities.optimization_algorithms(SCI_optimizer,cnn, SCI_LR, SCI_SGD_MOMENTUM,
                                                       SCI_REGULARIZATION)
        
        if GPU_SELECT == 2:
            if torch.cuda.device_count() > 1:
                cnn = nn.DataParallel(cnn,device_ids=[0, 1], dim = 0) 
            cnn = cnn.cuda()                
        if GPU_SELECT == 1:
            cnn.to(device)  
        if GPU_SELECT == 0:
            cnn.to(device)        

            
           
        cnn.apply(CNN6.weights_init2)    
        #cnn.apply(CNN6.weights_reset)        
        cnn.share_memory()
     
        loss_func = nn.CrossEntropyLoss()

        def create_loss(LOSS):   
            print('*** LOSS ******:',  LOSS)
            if LOSS == 1:
                loss_func = nn.BCELoss()
                print('*********  BCELoss')
            if LOSS == 2:
                loss_func = nn.MultiMarginLoss()
                print('*********  MMLoss')                               
            if LOSS == 3:
                loss_func = nn.CrossEntropyLoss() 
                print('********* CrossEntropyLoss ')                 
            return loss_func
        

        
        MM = float(str(SCI_MM))

        LR = float(str(SCI_LR))
        train_losses = []         # to track the training loss as the model trains
        output = 0
        loss = 0
        accuracy = 0
        early_stopping.counter = 0
        early_stopping.best_score = None
        early_stopping.early_stop = False
        early_stopping.verbose = False  
        TEST_RESULTS = torch.zeros(1,2)
    
        loss_type = create_loss(SCI_loss_type)
        
        #cnn, optimizer = amp.initialize(
        #   cnn, optimizer, opt_level=BITS, 
        #   keep_batchnorm_fp32=True, loss_scale="dynamic"
        #)
    
        Utillities.listing(optimizer, SCI_SGD_MOMENTUM, SCI_BN_MOMENTUM, 
                           SCI_L_SECOND, SCI_LR, SCI_RELU, 
                           SCI_BIAS, SCI_loss_type, SCI_REGULARIZATION, 
                           SCI_BATCH_SIZE, SCI_DROPOUT, SCI_LAST_LAYER, SCI_ACT_LAYER)

    
        # Data Loader for easy mini-batch return in training
        SCI_BATCH_SIZE = int(SCI_BATCH_SIZE)
        train_loader = Data.DataLoader(dataset = dataset.train_dataset, batch_size = SCI_BATCH_SIZE, shuffle = True, num_workers = 0, drop_last=True, pin_memory=True)
        validation_loader = Data.DataLoader(dataset = dataset.validation_dataset, batch_size = 24, shuffle = True, num_workers = 0, drop_last=True, pin_memory=True)    
        test_loader = Data.DataLoader(dataset = dataset.test_dataset, batch_size = 600, shuffle = True, num_workers = 0, drop_last=True, pin_memory=True)
    
        for epoch in range(SCI_EPOCHS):
            loss = None        
            cnn.train().cuda()
            for step, (train_data, train_target) in enumerate(train_loader):   
                train_data, train_target = train_data.to(device), train_target.to(device)
                output = cnn(train_data)                # forward pass: compute predicted outputs by passing inputs to the model     
                loss = loss_func(output, train_target)
                train_losses.append(loss.item())
                #batch_loss.backward()
                loss.backward()                               # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.zero_grad()
                optimizer.step()                              # perform a single optimization step (parameter update)
      
            cnn.eval().cuda()                 # switch to evaluation (no change) mode           
            valid_loss = 0
            accuracy = 0
            running_loss = 0.0
            with torch.no_grad():
                for step, (validation_data, validation_target) in enumerate(validation_loader):
                    validation_data, validation_target = validation_data.to(device), validation_target.to(device)
                    output = cnn(validation_data)      
                    valid_loss = loss_func(output, validation_target)
                    running_loss += valid_loss.item()
                epoch_val_loss = running_loss / len(validation_loader)    
                #if epoch % 100 == 0: 
                #    print('validation loss:',epoch_val_loss)                  
            train_losses = []
            early_stopping(epoch_val_loss, cnn)
        
            if early_stopping.early_stop:
                if os.path.exists('checkpoint.pt'):
                    #cnn = TheModelClass(*args, **kwargs)
                    print("Loaded the model with the lowest Validation Loss!")
                    cnn.load_state_dict(torch.load('checkpoint.pt'))  # Choose whatever GPU device number you want
                    cnn.to(device)
                break
            running_loss = 0.0
        cnn.eval()
        class_correct = list(0. for i in range(1000))
        class_total = list(0. for i in range(1000))
        with torch.no_grad():
            for (test_data, test_target) in test_loader:
                test_data, test_target = test_data.to(device), test_target.to(device)
                outputs = cnn(test_data)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == test_target).squeeze()
                dx = ((c.cpu()).numpy()).astype(int)
                #dx = 600
                for i in range(test_target.size(0)):
                    label = test_target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(dataset.CLASSES):
            TEST_RESULTS[0,i] = class_correct[i] / dataset.TESTED_ELEMENTS[i]
            print('Class: ',i,' accuracy: ', TEST_RESULTS[0,i])   
            print('Class: ',i,' correct: ', class_correct[i],' of ',dataset.TESTED_ELEMENTS[i])

        #mp.matshow(dx.reshape((20, 30)))
        #mp.ylabel('Correct Results')
        #mp.colorbar()
        #mp.show()
        percent = (TEST_RESULTS[0,0]+TEST_RESULTS[0,1])/2
        print('Final percentage: ',percent)
        CreditCost = 0
        CreditCost = int((1 - TEST_RESULTS[0,0]) * dataset.TESTED_ELEMENTS[0] + (1 - TEST_RESULTS[0,1]) * dataset.TESTED_ELEMENTS[1] * 5)
    
        if TEST_RESULTS[0,0] < 0.05 or TEST_RESULTS[0,1] < 0.05 :
            CreditCost = CreditCost + 300
    
        print('Last epoch: ', epoch)
    
        if os.path.exists('checkpoint.pt'):  
            os.remove('checkpoint.pt') 

        print()
        print()
        #print('Credit Cost: ',CreditCost)
        CreditCost = CreditCost + (SCI_SGD_MOMENTUM + SCI_DROPOUT + SCI_BATCH_SIZE + SCI_L_SECOND + SCI_optimizer + SCI_loss_type+ SCI_LR+ SCI_BN_EPS+SCI_BN_STATS+SCI_LAST_LAYER+SCI_ACT_LAYER)/10000
        print('Credit Cost: ',CreditCost)
        
        if -CreditCost > MaxCredit : 
            MaxCredit = -CreditCost
        print('Best Score So Far: ',MaxCredit)   
        
        CreditVector[count] = MaxCredit    
        CreditVec[count] = count
        # plot the data
        #fig = mp.figure()
        #ax = fig.add_subplot(1, 1, 1)
        #ax.plot(CreditVec, -CreditVector, color='tab:orange')
        #print(CreditVec, -CreditVector)
        count = count + 1
        # display the plot
        #mp.show()       
        return CreditCost
    
    
    optimizer = BayesianOptimization(
        f=objective,
        #pbounds=pbounds,
        pbounds={'SCI_RELU': (1,1.99), 
                 'SCI_BIAS': (1,1.99), 
                 'SCI_loss_type': (2, 2.99), 
                 'SCI_optimizer': (9, 9.99),
                 'SCI_LR': (0.00002, 0.009), 
                 'SCI_MM': (0.001, 0.999), 
                 'SCI_REGULARIZATION': (0, 0.8), 
                 'SCI_EPOCHS': (100, 200), 
                 'SCI_BATCH_SIZE': (90, 210), 
                 'SCI_DROPOUT': (0.5, 0.7), 
                 'SCI_L_SECOND': (64, 196), 
                 'SCI_BN_MOMENTUM': (0.1, 0.1), 
                 'SCI_SGD_MOMENTUM': (0, 0.999), 
                 'SCI_BN_EPS':(0,9.99),
                 'SCI_BN_STATS':(0,0.99),
                 'SCI_LAST_LAYER':(1,1.99),
                 'SCI_ACT_LAYER':(2,2.99)
                },
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
        

    #optimizer.maximize(
        #n_iter=TRIALS, acq="ucb", kappa=0.1
    #)
    
    
    optimizer.maximize(
        init_points = RANDOM_STARTS,
        n_iter = TRIALS,
        #acq="ucb", kappa=0.1
        
        acq="ei", xi=1e-4
    )
    
    
    print(optimizer.max)
    
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))


end.record()

#print('Minimum Credit Cost: ',Min_Credit_Cost)

print()
print('Total execution time (minutes): ',start.elapsed_time(end)/60000)

torch.cuda.empty_cache()

if GET_STATS:
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


# In[ ]:




