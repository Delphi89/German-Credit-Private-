#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
        
        

