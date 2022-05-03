#!/usr/bin/env python
# coding: utf-8

# ### BASE

# Title
# 
# ### Title
# 
# # Title
# 
# ## Title
# 
# #### Title
# 
# ##### Title
# 
# ###### Title

# In[2]:


print('Hello, world')

#shift + enter, control + enter


# In[ ]:


# esc + H СПИСОК ГОРЯЧИХ КЛАВИШ
# tab ПОДСКАЗКА
# tab + shift ПОДСКАЗКА С ДОКУМЕНТАЦИЕЙ


# In[3]:


get_ipython().run_line_magic('pinfo', 'print')

#shift + enter


# In[4]:


get_ipython().system('dir')


# In[5]:


get_ipython().run_line_magic('lsmagic', '')


# In[7]:


get_ipython().run_line_magic('run', 'script1.py')


# In[8]:


get_ipython().run_line_magic('load', 'script1.py')


# In[11]:


import numpy as np

a = [i for i in range(10)]
a


# In[12]:


b = np.array(a)
b


# In[13]:


type(b)


# In[15]:


b = np.array(a, dtype = float)


# In[16]:


a = [0, 1, 2, 3, 4, 5, 6, 7, 8.8]


# In[18]:


b = np.array(a, dtype = int)
b


# In[20]:


b = np.array(a, dtype = float)
b


# In[21]:


b = np.array(a, dtype = str)
b


# In[22]:


b.dtype


# In[23]:


b = np.array(a, dtype = float)
b


# In[24]:


b.dtype


# In[25]:


b[0]


# In[26]:


b[1]


# In[27]:


b[2]


# In[28]:


b[3]


# In[31]:


b.ndim

# ПОСМОТРЕТЬ РАЗМЕРНОСТЬ МАССИВА


# In[32]:


b.shape

# ПОСМОТРЕТЬ РАЗМЕРНОСТЬ МАССИВА


# ### 2D ARRAY

# In[37]:


b = np.array([[0, 1, 2, 3, 4, 5, 6, 7], 
             [9, 10, 11, 12, 13, 14, 15, 16], 
             [17, 18, 19, 20, 21, 22, 23, 24]])
b


# In[38]:


b.shape

# ПОСМОТРЕТЬ РАЗМЕРНОСТЬ МАССИВА


# In[39]:


b.size

# ОБЩЕЕ КОЛЛИЧЕСТВО ЭЛЕМЕНТОВ


# In[40]:


b


# In[41]:


b[2, 5]


# In[42]:


b[0][0]


# In[43]:


b[0, -1]


# In[44]:


b[0:2, 0:3]


# In[45]:


b[:2, :3]


# In[46]:


b[1:, 5:]


# In[47]:


c = b[1:, 5:]
c


# In[49]:


c[1, 2] = 888
c


# In[50]:


b


# ### LIN.AL - 1

# In[51]:


import numpy as np


# In[52]:


a = np.array([0, 1, 2, 3, 4])
b = np.array([5, 6, 7, 8, 9])


# In[54]:


c = a + b
c


# In[55]:


c  = np.add(a, b)
c


# In[56]:


a = np.array([9, 7, 2, 4, 8])
b = np.array([6, 5, 1, 8, 3])


# In[57]:


d = a - b
d


# In[58]:


d = np.subtract(a, b)
d


# In[59]:


c = a * 2
c


# In[60]:


c = -10 * a
c


# In[62]:


c = a * 0.5
c


# In[63]:


a.dot(-10)


# In[64]:


np.dot(-10, a)


# In[65]:


np.multiply(-10, a)


# In[66]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])


# In[67]:


a.shape


# In[68]:


b.shape


# In[69]:


sp = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
sp


# In[70]:


a @ b


# In[72]:


sp = np.dot(a, b)
sp


# In[73]:


A = np.array([[0, 1], [2,3], [4,5]])


# In[74]:


B = np.array([[6,7], [8, 9], [10, 11]])


# In[75]:


A.shape


# In[76]:


B.shape


# In[77]:


C = A + B
C


# In[78]:


C = np.add(A, B)
C


# In[79]:


D = B - A
D


# In[80]:


D = np.subtract(B, A)
D


# In[81]:


S = A * 3
S


# In[82]:


S = np.dot(A, 3)
S


# ### LIN.AL - 2

# In[83]:


import numpy as np


# In[84]:


x1 = np.array([[1,2], [3,4], [5,6]])


# In[85]:


x2 = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])


# In[86]:


x1.shape[1] == x2.shape[0]


# In[87]:


z = np.dot(x1, x2)
z


# In[88]:


z.shape


# In[89]:


z[0, 0]


# In[90]:


z.size


# In[91]:


A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


# In[92]:


A_square = np.dot(A, A)
A_square


# In[93]:


A_square = np.linalg.matrix_power(A, 2)
A_square


# In[94]:


I = np.eye(3)
I


# In[95]:


A = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])


# In[96]:


At = A.transpose()
At


# In[97]:


At = A.T
At


# In[100]:


A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


# In[101]:


np.linalg.det(A)


# In[102]:


np.linalg.matrix_rank(A)


# In[103]:


# Обратная матрица
B = np.array([[7, 8, 6], [5, 9, 2], [3, 4, 7]])


# In[104]:


B_inv = np.linalg.inv(B)
B_inv


# In[105]:


B_inv2 = np.linalg.inv(B_inv)
B_inv2


# ### MASSIV

# In[106]:


import numpy as np


# In[115]:


a = np.zeros((3, 4))
a


# In[108]:


a = np.ones((5, 3))
a


# In[110]:


np.arange(10)


# In[111]:


np.arange(0, 10)


# In[112]:


np.arange(0, 10, 2)


# In[114]:


np.arange(10, 0, -1)


# In[ ]:


np.arange(10, 0, 0.1)


# In[116]:


np.linspace(0, 9.9, 100)


# In[117]:


np.logspace(0, 3, 4)


# In[119]:


np.random.sample()


# In[120]:


np.random.sample(3)


# In[121]:


np.random.sample((2, 3))


# In[122]:


np.random.sample((48, 63, 3))


# In[124]:


np.random.randn(10)


# In[125]:


np.random.randn(3, 4)


# In[126]:


np.random.randint(0, 100, 10)


# In[129]:


np.random.randint(0, 10, (3, 4))


# In[130]:


b = np.array([1, 5, 2, 7, 3, 8, 6, 4])


# In[132]:


np.random.choice(b)


# In[133]:


np.random.choice(b, 3)


# In[135]:


a = np.arange(12)
a


# In[136]:


a.reshape(3, 4)


# In[137]:


np.reshape(a, (3, -1))


# In[138]:


np.reshape(a, (-1, 4))


# In[140]:


a.resize(3, 4)
a


# In[141]:


a = a.flatten()
a


# In[142]:


a = np.zeros((2, 3))
a


# In[143]:


b = np.ones((2, 3))
b


# In[145]:


v = np.vstack((a, b))
v


# In[147]:


a.shape, b.shape, v.shape


# In[151]:


np.concatenate((a, b), axis = 0)


# In[152]:


h = np.hstack((a, b))
h


# In[153]:


a.shape, b.shape, v.shape


# In[154]:


np.concatenate((a, b), axis = 1)


# In[156]:


d = np.dstack([a, b])
d


# ### FUNCTIONS

# In[157]:


import numpy as np


# In[159]:


a = np.random.randint(0, 20, 10)
a


# In[160]:


a[a > 10]


# In[161]:


a[(a > 10) & (a % 2 == 0)]


# In[162]:


a[(a > 10) | (a % 3 == 0)]


# In[163]:


a


# In[164]:


np.where(a > 10)


# In[165]:


a = [1, 4, 7]
b = [9, 2, 5]

np.where([True, False, True], a, b)


# In[166]:


a = np.array([[1, 3, 9, 9], [5, 0, 1, 5], [2, 7, 3, 5]])


# In[168]:


a[[0, 2, 1], :]


# In[169]:


a.argsort(axis = 0)


# In[170]:


a[a[:, 0].argsort(), :]


# In[171]:


a = np.arange(10)
a


# In[172]:


np.random.shuffle(a)


# In[173]:


a


# In[174]:


a = np.arange(1, 7).reshape(2, 3)
a


# In[175]:


np.log(a)


# In[176]:


np.exp(a)


# In[177]:


np.sum(a)


# In[178]:


a.sum(axis = 0)


# In[179]:


a.sum(axis = 1)


# In[180]:


a


# In[181]:


10 + a


# In[182]:


a = np.arange(12).reshape(3, -1)
a


# In[183]:


b = np.arange(12, 24).reshape(-1, 4)
b


# In[184]:


a * b


# In[185]:


a = np.array([[2, 5], [3, 4], [6, 1]])


# In[186]:


b = np.array([1, 2])


# In[187]:


a + b


# In[188]:


a = np.random.randint(0, 12, (3, 4))
a


# In[189]:


a.min()


# In[190]:


a.min(axis = 0)


# In[191]:


a.max()


# In[192]:


a.max(axis = 1)


# In[193]:


a.mean()


# In[194]:


a.mean(axis = 0)


# In[195]:


a.std(axis = 0)

