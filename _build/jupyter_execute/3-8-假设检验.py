#!/usr/bin/env python
# coding: utf-8

# # 第 8 节　假设检验
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学
# 
# 

# ### 13. t 检验的实现：环境准备

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


# In[2]:


# 读入数据
junk_food = pd.read_csv(
    "3-8-1-junk-food-weight.csv")["weight"]
junk_food.head()


# ### 14. t 检验的实现：计算 t 值

# In[3]:


# 样本均值
mu = sp.mean(junk_food)
mu


# In[4]:


# 自由度
df = len(junk_food) - 1
df


# In[5]:


# 标准误差
sigma = sp.std(junk_food, ddof = 1)
se = sigma / sp.sqrt(len(junk_food))
se


# In[6]:


# t 值
t_value = (mu - 50) / se
t_value


# ### 15. t 检验的实现：计算 p 值

# In[7]:


# p 值
alpha = stats.t.cdf(t_value, df = df)
(1 - alpha) * 2


# In[8]:


# t 检验
stats.ttest_1samp(junk_food, 50)


# ### 16. 通过模拟实验计算 p 值

# In[9]:


# 样本的相关信息 (一部分)
size = len(junk_food)
sigma = sp.std(junk_food, ddof = 1)


# In[10]:


# 存放 t 值的窗口
t_value_array = np.zeros(50000)


# In[11]:


# 总体均值为 50, 以接受零假设为前提进行 50,000 次抽样并计算 t 值的实验
np.random.seed(1)
norm_dist = stats.norm(loc = 50, scale = sigma)
for i in range(0, 50000):
    sample = norm_dist.rvs(size = size)
    sample_mean = sp.mean(sample)
    sample_std = sp.std(sample, ddof = 1)
    sample_se = sample_std / sp.sqrt(size)
    t_value_array[i] = (sample_mean - 50) / sample_se


# In[12]:


(sum(t_value_array > t_value) / 50000) * 2


# In[ ]:




