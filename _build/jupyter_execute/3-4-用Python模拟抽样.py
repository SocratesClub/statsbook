#!/usr/bin/env python
# coding: utf-8

# # 第 4 节　用 Python 模拟抽样
# 
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学
# 
# 

# ### 1. 环境准备

# In[1]:


# 用于数值计算的库
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

# 用于绘图的库
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# 设置浮点数打印精度
get_ipython().run_line_magic('precision', '3')
# 在 Jupyter Notebook 里显示图形
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 3. 在只有 5 条鱼的湖中抽样

# In[2]:


fish_5 = np.array([2,3,4,5,6])
fish_5


# In[3]:


# 从总体中随机抽样
np.random.choice(fish_5, size = 1, replace = False)


# In[4]:


# 从总体中随机抽样
np.random.choice(fish_5, size = 3, replace = False)


# In[5]:


np.random.choice(fish_5, size = 3, replace = False)


# In[6]:


# 设定随机数种子以得到相同结果
np.random.seed(1)
np.random.choice(fish_5, size = 3, replace = False)


# In[7]:


np.random.seed(1)
np.random.choice(fish_5, size = 3, replace = False)


# In[8]:


# 计算样本均值
np.random.seed(1)
sp.mean(
    np.random.choice(fish_5, size = 3, replace = False)
)


# ### 6. 从鱼较多的湖中抽样

# In[9]:


# 鱼较多的总体
fish_100000 = pd.read_csv(
    "3-4-1-fish_length_100000.csv")["length"]
fish_100000.head()


# In[10]:


len(fish_100000)


# In[11]:


# 抽样模拟实验
sampling_result = np.random.choice(
    fish_100000, size = 10, replace = False)
sampling_result


# In[12]:


# 样本均值
sp.mean(sampling_result)


# ### 7. 总体分布

# In[13]:


sp.mean(fish_100000)


# In[14]:


sp.std(fish_100000, ddof = 0)


# In[15]:


sp.var(fish_100000, ddof = 0)


# In[16]:


sns.distplot(fish_100000, kde = False, color = 'black')


# ### 8. 对比总体分布和正态分布的概率密度函数

# In[17]:


x = np.arange(start = 1, stop = 7.1, step = 0.1)
x


# In[18]:


stats.norm.pdf(x = x, loc = 4, scale = 0.8)


# In[19]:


plt.plot(x, 
         stats.norm.pdf(x = x, loc = 4, scale = 0.8), 
         color = 'black')


# In[20]:


# 把正态分布的概率密度和总体的直方图重合
sns.distplot(fish_100000, kde = False, 
             norm_hist = True, color = 'black')
plt.plot(x, 
         stats.norm.pdf(x = x, loc = 4, scale = 0.8), 
         color = 'black')


# ### 9. 抽样过程的抽象描述

# In[21]:


sampling_norm = stats.norm.rvs(
    loc = 4, scale = 0.8, size = 10)
sampling_norm


# In[22]:


# 样本均值
sp.mean(sampling_norm)


# In[ ]:




