#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import mish as Mish
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)
   
class CNN6(nn.Module):
    
    def ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU):
        
        if int(SCI_ACT_LAYER) == 3:
            return (Swish())     
        if int(SCI_ACT_LAYER) == 2:
            return (nn.LeakyReLU(negative_slope=SCI_RELU))
        if int(SCI_ACT_LAYER) == 1:
            return (nn.ReLU(SCI_RELU))

                
    def LAST_LAYER(SCI_LAST_LAYER):
        
        if int(SCI_LAST_LAYER) == 5:
            return (Swish())
        if int(SCI_LAST_LAYER) == 3:
            return (nn.Sigmoid())    
        if int(SCI_LAST_LAYER) == 4:
            return (nn.Softmax(0))
        if int(SCI_LAST_LAYER) == 1:
            return (nn.LogSoftmax(0))
        if int(SCI_LAST_LAYER) == 2:
            return (Mish.Mish())              
            
       
    
    def __init__(self,L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES, LINEARITY, SCI_BN_EPS, SCI_BN_STATS, SCI_LAST_LAYER):
        super(CNN6, self).__init__()
         
        self.linear_1 = nn.Sequential(                  
            nn.BatchNorm2d(L_FIRST, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),
            nn.Conv2d(L_FIRST, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),           
        )
        
        self.linear_2 = nn.Sequential(                  
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),        
        )      
        
        self.linear_21 = nn.Sequential(                  
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
        self.linear_22 = nn.Sequential(   
            nn.Conv2d(L_FIRST, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),        
            
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
  
        self.linear_23 = nn.Sequential(      
            nn.Linear(SCI_L_SECOND, SCI_L_SECOND, bias=SCI_BIAS),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),       
        )

            
        self.linear_3 = nn.Sequential(
            nn.Linear(SCI_L_SECOND, CLASSES, bias=SCI_BIAS),
            CNN6.LAST_LAYER(SCI_LAST_LAYER)
            #nn.Sigmoid()
            #nn.LogSoftmax(0)
        )
        

    def forward(self, x): 
        x1 = self.linear_1(x) 
        x2 = self.linear_2(x1) + self.linear_21(x1) + self.linear_22(x)
        x3 = x2.view(x2.size(0), -1)
        x4 = x1.view(x1.size(0), -1)
        x5 = self.linear_23(x4) 
        x6 = x3 + x5
        output = self.linear_3(x6)
        return output, x   
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)        
          
    def weights_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()  