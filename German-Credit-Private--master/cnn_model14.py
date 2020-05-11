#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import mish as Mish
import torch.nn as nn
import torch


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
        
        if int(SCI_LAST_LAYER) == 7:
            return (Swish())
        if int(SCI_LAST_LAYER) == 2:
            return (nn.Sigmoid())    
        if int(SCI_LAST_LAYER) == 8:
            return (nn.Softmax(0))
        if int(SCI_LAST_LAYER) == 1:
            return (nn.LogSoftmax(0))
        if int(SCI_LAST_LAYER) == 6:
            return (Mish.Mish())              
        if int(SCI_LAST_LAYER) == 3:
            return (nn.Softsign())     
        if int(SCI_LAST_LAYER) == 4:
            return (nn.Hardtanh())   
        if int(SCI_LAST_LAYER) == 5:
            return (nn.Tanh())                       
       
    
    def __init__(self, L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES, SCI_BN_EPS, SCI_BN_STATS, SCI_LAST_LAYER, SCI_ACT_LAYER):
        super(CNN6, self).__init__()
         
        self.linear_1 = nn.Sequential(                  
            nn.BatchNorm2d(L_FIRST, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),
            nn.Conv2d(L_FIRST, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),           
        )
        
        self.linear_2 = nn.Sequential(                  
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS), 
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
        )      
        
        self.linear_21 = nn.Sequential(                  
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
            nn.Dropout(p = SCI_DROPOUT),
        )
        
        self.linear_22 = nn.Sequential(   
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU), 
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),        
            
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
            nn.Dropout(p = SCI_DROPOUT),
        )

        self.linear_23 = nn.Sequential(   
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU), 
            nn.BatchNorm2d(SCI_L_SECOND, eps=SCI_BN_EPS, momentum=SCI_BN_MOMENTUM, affine=True, track_running_stats=SCI_BN_STATS),        
            
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
            nn.Dropout(p = SCI_DROPOUT),
            
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
            nn.Dropout(p = SCI_DROPOUT),
            
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
            nn.Dropout(p = SCI_DROPOUT),
            
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            CNN6.ACTIVATION_LAYER(SCI_ACT_LAYER, SCI_RELU),
            nn.Dropout(p = SCI_DROPOUT),
                        
        )
                   
        self.linear_3 = nn.Sequential(
            nn.Linear(SCI_L_SECOND, CLASSES, bias=SCI_BIAS),
            CNN6.LAST_LAYER(SCI_LAST_LAYER)
        )   

    def forward(self, x): 
        x1 = self.linear_1(x) 
        x2 = self.linear_2(x1) + self.linear_21(x1) + x1
        #x3 = self.linear_22(x2) + self.linear_23(x2) + x2
        #x4 = self.linear_2(x3) + self.linear_21(x3) + x3 
        #x5 = x2 #x4 + x3 + x2 + x1
        x3 = x2.view(x2.size(0), -1)
        output = self.linear_3(x3)
        return output 
    
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)                
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)       
                
    def weights_init2(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0, 0.01)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 0, 0.01)
            nn.init.constant_(m.bias.data, 0)                   
          
    def weights_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            m.reset_parameters()  