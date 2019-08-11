#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch.optim as optim
from adamw import AdamW


class Utillities:

    def listing(optimizer, SCI_SGD_MOMENTUM, SCI_BN_MOMENTUM, SCI_L_SECOND, SCI_LR, SCI_RELU, SCI_BIAS, SCI_loss_type, REGULARIZATION, SCI_BATCH_SIZE, SCI_DROPOUT, SCI_LINEARITY):
        print('Optimization: ', optimizer)
        if optimizer == 'SGD':
            print('MM: ',SCI_SGD_MOMENTUM)
        print('Batch Normalization Momentum: ',SCI_BN_MOMENTUM)   
        print('Nodes: ', SCI_L_SECOND)         
        print('LR: ', SCI_LR)         
        print('RELU: ', SCI_RELU)       
        print('BIAS: ', SCI_BIAS)   
        print('Loss Type: ', SCI_loss_type)   
        print('REGULARIZATION: ', REGULARIZATION)    
        print('BATCH_SIZE: ', SCI_BATCH_SIZE)
        print('Dropout: ', SCI_DROPOUT)
        print('Final Linear Layers: ', SCI_LINEARITY)
        
 

    def optimization_algorithms(SCI_optimizer,cnn, LR, SCI_SGD_MOMENTUM, REGULARIZATION):  

        print('SCI_optimizer: ', SCI_optimizer)
        print(type(SCI_optimizer))        
        
        if (SCI_optimizer == 'Adam') or (int(SCI_optimizer) == 1) :
            optimizer = optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=REGULARIZATION)
        if (SCI_optimizer == 'AMSGrad') or (int(SCI_optimizer) == 2) :
            optimizer = optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=REGULARIZATION, amsgrad=True)
        if (SCI_optimizer == 'AdamW') or (int(SCI_optimizer) == 3) :
            optimizer = AdamW(cnn.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay = REGULARIZATION)            
        #if (SCI_optimizer == 'SparseAdam') or (int(SCI_optimizer) == 4) :
            #optimizer = optim.SparseAdam(cnn.parameters(), lr=LR)            
        if (SCI_optimizer == 'SGD') or (int(SCI_optimizer) == 5) :
            optimizer = optim.SGD(cnn.parameters(), lr=LR, momentum=SCI_SGD_MOMENTUM, weight_decay=REGULARIZATION)
        if (SCI_optimizer == 'Adadelta') or (int(SCI_optimizer) == 6) :
            optimizer = optim.Adadelta(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION)
        #if (SCI_optimizer == 'Adagrad') or (int(SCI_optimizer) == 7) :
        #    optimizer = optim.Adagrad(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION)
        if (SCI_optimizer == 'Adamax') or (int(SCI_optimizer) == 8) :
            optimizer = optim.Adamax(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION)   
        if (SCI_optimizer == 'ASGD') or (int(SCI_optimizer) == 9) :
            optimizer = optim.ASGD(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION)               
        #if (SCI_optimizer == 'LBFGS') or (int(SCI_optimizer) == 10) :
            #optimizer = optim.LBFGS(cnn.parameters(), lr=LR)         
        if (SCI_optimizer == 'RMSprop') or (int(SCI_optimizer) == 4) :
            optimizer = optim.RMSprop(cnn.parameters(), lr=LR)  
        if (SCI_optimizer == 'Rprop') or (int(SCI_optimizer) == 7) :
            optimizer = optim.Rprop(cnn.parameters(), lr=LR)              
        
      
        return optimizer
    
    
        