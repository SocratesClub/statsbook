#!/usr/bin/env python
# coding: utf-8

# # 第 1 节　使用 Python 进行描述统计：单变量
# 
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学

# ### 1. 统计分析与 scipy

# In[1]:


# 用于数值计算的库
import numpy as np
import scipy as sp

# 设置浮点数打印精度
get_ipython().run_line_magic('precision', '3')


# ### 2. 单变量数据的操作

# In[2]:


fish_data = np.array([2,3,3,4,4,4,4,5,5,6])
fish_data


# ### 3. 实现：总和与样本容量

# In[3]:


# 总和
sp.sum(fish_data)


# In[4]:


# 参考
np.sum(fish_data)


# In[5]:


# 参考
fish_data.sum()


# In[6]:


# 参考
sum(fish_data)


# In[7]:


# 样本容量
len(fish_data)


# ### 4. 实现：均值（期望值）

# In[8]:


# 计算均值
N = len(fish_data)
sum_value = sp.sum(fish_data)
mu = sum_value / N
mu


# In[9]:


# 计算均值的函数
sp.mean(fish_data)


# ### 5. 实现：样本方差

# In[10]:


# 样本方差
sigma_2_sample = sp.sum((fish_data - mu) ** 2) / N
sigma_2_sample


# In[11]:


fish_data


# In[12]:


fish_data - mu


# In[13]:


(fish_data - mu) ** 2


# In[14]:


sp.sum((fish_data - mu) ** 2)


# In[15]:


# 计算样本方差的函数
sp.var(fish_data, ddof = 0)


# ### 6. 实现：无偏方差

# In[16]:


# 无偏方差
sigma_2 = sp.sum((fish_data - mu) ** 2) / (N - 1)
sigma_2


# In[17]:


# 无偏方差
sp.var(fish_data, ddof = 1)


# ### 7. 实现：标准差

# In[18]:


# 标准差
sigma = sp.sqrt(sigma_2)
sigma


# In[19]:


# 计算标准差的函数
sp.std(fish_data, ddof = 1)


# ### 8. 补充：标准化

# In[20]:


fish_data - mu


# In[21]:


sp.mean(fish_data - mu)


# In[22]:


fish_data / sigma


# In[23]:


sp.std(fish_data / sigma, ddof = 1)


# In[24]:


standard = (fish_data - mu) / sigma
standard


# In[25]:


sp.mean(standard)


# In[26]:


sp.std(standard, ddof = 1)


# ### 9. 补充：其他统计量

# In[27]:


# 最大值
sp.amax(fish_data)


# In[28]:


# 最小值
sp.amin(fish_data)


# In[29]:


# 中位数
sp.median(fish_data)


# In[30]:


fish_data_2 = np.array([2,3,3,4,4,4,4,5,5,100])


# In[31]:


sp.mean(fish_data_2)


# In[32]:


sp.median(fish_data_2)


# ### 10. 实现：scipy.stats 与四分位数

# In[33]:


from scipy import stats


# In[34]:


fish_data_3 = np.array([1,2,3,4,5,6,7,8,9])
stats.scoreatpercentile(fish_data_3, 25)


# In[35]:


stats.scoreatpercentile(fish_data_3, 75)

