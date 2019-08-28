#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch.nn as nn

class CNN1(nn.Module):
       
    def __init__(self,L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES):
        super(CNN1, self).__init__()
        
        self.L_FIRST = L_FIRST
        self.SCI_L_SECOND = SCI_L_SECOND
        self.KERNEL_X = KERNEL_X
        self.SCI_BIAS = SCI_BIAS
        self.SCI_BN_MOMENTUM = SCI_BN_MOMENTUM
        self.SCI_RELU = SCI_RELU
        self.SCI_DROPOUT = SCI_DROPOUT
        self.CLASSES = CLASSES
                                             
        self.linear_1 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),               
            nn.ReLU(SCI_RELU), 
        )
        
        self.linear_2 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(int(KERNEL_X/2), 1), stride=2, padding=(3,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU), 
        )
            
        self.linear_21 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.Dropout(p = SCI_DROPOUT),
        )
                
        self.linear_22= nn.Sequential(                  
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(int(KERNEL_X/4), 1), stride=2, padding=(2,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU), 
        )                
        
        self.linear_3 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),                
            nn.ReLU(SCI_RELU), 
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=2, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU), 
        )  
        
        self.linear_4 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(int(KERNEL_X/4), 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU), 
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(1,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU),           
        )        

        self.linear_5 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(int(KERNEL_X/4), 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU), 
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(int(KERNEL_X/4), 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU),    
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(int(KERNEL_X/4), 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU),     
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(int(KERNEL_X/4), 1), stride=2, padding=(0,0), bias=SCI_BIAS),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            nn.ReLU(SCI_RELU),                   
        )           
            
        self.linear_7 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(int(KERNEL_X/8), 1), stride=2, padding=(1,0), bias=SCI_BIAS),
            nn.MaxPool2d(kernel_size=(int(KERNEL_X/2), 1), stride=1),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),               
            nn.ReLU(SCI_RELU), 
        )
                
        self.linear_71 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(int(KERNEL_X/6), 1), stride=2, padding=(1,0), bias=SCI_BIAS),
            nn.AvgPool2d(kernel_size=(int(KERNEL_X/2), 1), stride=1),
            nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),               
            nn.ReLU(SCI_RELU), 
        )
                                     
        self.linear_8 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(int(KERNEL_X/8), 1), stride=2, padding=(1,0), bias=SCI_BIAS),
            nn.Dropout(p = SCI_DROPOUT),
            nn.ReLU(SCI_RELU), 
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(int(KERNEL_X/8), 1), stride=2, padding=(1,0), bias=SCI_BIAS),
            nn.Dropout(p = SCI_DROPOUT),
            nn.ReLU(SCI_RELU), 
        )
 
        self.linear_6 = nn.Sequential(      
            nn.Linear(SCI_L_SECOND*8, SCI_L_SECOND, bias=SCI_BIAS),
            nn.Dropout(p = SCI_DROPOUT),                
            nn.ReLU(SCI_RELU), 
            nn.Linear(SCI_L_SECOND, CLASSES, bias=SCI_BIAS),
            nn.Softmax(1)
        )                    
                         
    def forward(self, x): 
        x1  = self.linear_1(x)
        x2  = self.linear_2(x)
        x21 = self.linear_21(x)
        x3  = self.linear_3(x)
        x4  = self.linear_4(x)
        x5  = self.linear_5(x)        
        x7  = self.linear_7(x)  
        x71 = self.linear_71(x)               
        x8  = self.linear_8(x)
        x10  = x1 + x2 + x21 + x3 + x4 
        x11  = x5 + x7 + x71 + x8
        x12 = self.linear_22(x10)  
        x13 = x11 + x12
        x9  = x13.view(x13.size(0), -1)
        output = self.linear_6(x9)
        return output, x



    
class CNN2(nn.Module):
    def __init__(self,L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES):
        super(CNN2, self).__init__()
        
        self.L_FIRST = L_FIRST
        self.SCI_L_SECOND = SCI_L_SECOND
        self.KERNEL_X = KERNEL_X
        self.SCI_BIAS = SCI_BIAS
        self.SCI_BN_MOMENTUM = SCI_BN_MOMENTUM
        self.SCI_RELU = SCI_RELU
        self.SCI_DROPOUT = SCI_DROPOUT
        self.CLASSES = CLASSES
              
        self.linear_1 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
        self.linear_2 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=2, padding=(1,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
        self.linear_3 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(6,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(6,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )  
        
        self.linear_4 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(12, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(12, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),            
        )        
        
      
        self.linear_5 = nn.Sequential(      
            nn.Linear(SCI_L_SECOND*2, SCI_L_SECOND, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
            nn.Linear(SCI_L_SECOND, CLASSES, bias=SCI_BIAS),
            nn.Softmax(1)
        ) 
                  
        
    def forward(self, x): 
        x1  = self.linear_1(x)
        x2  = self.linear_2(x)
        x3  = self.linear_3(x)
        x4  = self.linear_4(x)
        x5  = x1 + x2 + x3 + x4
        x9  = x5.view(x5.size(0), -1)
        output = self.linear_5(x9)
        return output, x
    
    
    
class CNN3(nn.Module):
    def __init__(self,L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES):
        super(CNN3, self).__init__()
        
        self.L_FIRST = L_FIRST
        self.SCI_L_SECOND = SCI_L_SECOND
        self.KERNEL_X = KERNEL_X
        self.SCI_BIAS = SCI_BIAS
        self.SCI_BN_MOMENTUM = SCI_BN_MOMENTUM
        self.SCI_RELU = SCI_RELU
        self.SCI_DROPOUT = SCI_DROPOUT
        self.CLASSES = CLASSES        
         
        self.linear_1 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
        self.linear_2 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(0,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
        self.linear_3 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(2,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,kernel_size=(KERNEL_X, 1), stride=1, padding=(1,0), bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )     
        
        self.linear_5 = nn.Sequential(      
            nn.Linear(SCI_L_SECOND*18, 64, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
            nn.Linear(64, 2, bias=SCI_BIAS),
            nn.Softmax(1)
        ) 
               
        self.drop_1 = nn.Sequential(      
            nn.Dropout(p = SCI_DROPOUT),
        )          
        
    def forward(self, x): 
        x1  = self.linear_1(x)
        x2  = self.linear_2(x)
        x3  = self.linear_3(x)
        x4  = x1 + x2 + x3
        x5 = x4.view(x4.size(0), -1)
        output = self.linear_5(x5)
        return output, x
    
    
class CNN4(nn.Module):
    def __init__(self,L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES):
        super(CNN4, self).__init__()
         
        self.linear_1 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )
            
        self.linear_2 = nn.Sequential(      
            nn.Linear(SCI_L_SECOND, SCI_L_SECOND, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
            nn.Linear(SCI_L_SECOND, CLASSES, bias=SCI_BIAS),
            nn.LogSoftmax(1)
        )  
        

    def forward(self, x): 
        x1 = self.linear_1(x)
        x2 = x1.view(x1.size(0), -1)
        output = self.linear_2(x2)
        return output, x    
    
    
class CNN5(nn.Module):
    def __init__(self,L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES):
        super(CNN5, self).__init__()
         
        self.linear_1 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        ).to('cuda:1')
            
        self.linear_2 = nn.Sequential(      
            nn.Linear(SCI_L_SECOND, SCI_L_SECOND, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
            nn.Linear(SCI_L_SECOND, SCI_L_SECOND, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),            
            nn.Linear(SCI_L_SECOND, SCI_L_SECOND, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),            
            nn.Linear(SCI_L_SECOND, CLASSES, bias=SCI_BIAS),
            nn.LogSoftmax(1)
        ).to('cuda:1')
        

    def forward(self, x): 
        x1 = self.linear_1(x)
        x2 = x1.view(x1.size(0), -1)
        output = self.linear_2(x2)
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
    
    
    
    
class CNN6(nn.Module):
    def __init__(self,L_FIRST, SCI_L_SECOND, KERNEL_X, SCI_BIAS, SCI_BN_MOMENTUM, SCI_RELU, SCI_DROPOUT, CLASSES, LINEARITY):
        super(CNN6, self).__init__()
         
        self.linear_1 = nn.Sequential(                  
            nn.Conv2d(L_FIRST, SCI_L_SECOND,1, bias=SCI_BIAS),
            #nn.ReLU(SCI_RELU), 
            #nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
        )
        
        self.linear_2 = nn.Sequential(                  
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            #nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
        )      
        
        self.linear_21 = nn.Sequential(                  
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
        self.linear_22 = nn.Sequential(   
            nn.Conv2d(L_FIRST, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            #nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
            
            nn.Conv2d(SCI_L_SECOND, SCI_L_SECOND,1, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU), 
            nn.Dropout(p = SCI_DROPOUT),
        )
        
  
        self.linear_23 = nn.Sequential(      
            nn.Linear(SCI_L_SECOND, SCI_L_SECOND, bias=SCI_BIAS),
            nn.ReLU(SCI_RELU),       
            #nn.BatchNorm2d(SCI_L_SECOND, momentum = SCI_BN_MOMENTUM),
        )

            
        self.linear_3 = nn.Sequential(
            nn.Linear(SCI_L_SECOND, CLASSES, bias=SCI_BIAS),
            nn.LogSoftmax(0)
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