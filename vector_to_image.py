#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

N = 20
Image = np.zeros((N,N))
print('original Image: \n',Image)

A = np.random.rand(1,N).astype(np.float64)
print('original matrix: \n',A)

B = normalize(A, axis=1, norm='l1') * 100
print('normalized matrix: \n',B)


# In[14]:


for j in range (N):
    for i in range (N):
        if ((B[0,j] / i) > 1) :
            Image[i,j] = 1
print('final image: \n',Image)
            


# In[15]:


plt.matshow(Image)
plt.ylabel('Credit Characteristics')
plt.colorbar()
plt.show()

