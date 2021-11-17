#!/usr/bin/env python
# coding: utf-8

# # 第 6 章　广义线性模型
# 
# ## 第 1 节　各种概率分布
# 
# 

# ### 8. 环境准备

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


# ### 9. 实现：二项分布

# In[2]:


# 二项分布的概率质量函数
sp.stats.binom.pmf(k = 1, n = 2, p = 0.5)


# In[3]:


# 从 N = 10, p = 0.5 的二项分布中生成随机数
np.random.seed(1)
sp.stats.binom.rvs(n = 10, p = 0.2, size = 5)


# In[4]:


# N = 10, p = 0.2 的二项分布
binomial = sp.stats.binom(n = 10, p = 0.2)

# 生成随机数
np.random.seed(1)
rvs_binomial = binomial.rvs(size = 10000)

# 概率质量函数
m = np.arange(0,10,1)
pmf_binomial = binomial.pmf(k = m)

# 绘制出样本直方图与概率质量函数
sns.distplot(rvs_binomial, bins = m, kde = False, 
             norm_hist = True, color = 'gray')
plt.plot(m, pmf_binomial, color = 'black')


# ### 14. 实现：泊松分布

# In[5]:


# 泊松分布的概率质量函数
sp.stats.poisson.pmf(k = 2, mu = 5)


# In[6]:


# 从 λ = 2 的泊松分布中生成随机数
np.random.seed(1)
sp.stats.poisson.rvs(mu = 2, size = 5)


# In[7]:


# λ = 2 的泊松分布
poisson = sp.stats.poisson(mu = 2)

# 生成随机数
np.random.seed(1)
rvs_poisson = poisson.rvs(size = 10000)

# 概率质量函数
pmf_poisson = poisson.pmf(k = m)

# 绘制样本直方图与概率质量函数
sns.distplot(rvs_poisson, bins = m, kde = False, 
             norm_hist = True, color = 'gray')
plt.plot(m, pmf_poisson, color = 'black')


# In[8]:


# N 非常大但 p 非常小的二项分布
N = 100000000
p = 0.00000002
binomial_2 = sp.stats.binom(n = N, p = p)

# 概率质量函数
pmf_binomial_2 = binomial_2.pmf(k = m)

# 绘制概率质量函数
plt.plot(m, pmf_poisson, color = 'gray')
plt.plot(m, pmf_binomial_2, color = 'black', 
         linestyle = 'dotted')


# In[ ]:




