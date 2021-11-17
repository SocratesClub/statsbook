#!/usr/bin/env python
# coding: utf-8

# # 第 7 节　参数估计
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学
# 
# 

# ### 2. 环境准备

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
fish = pd.read_csv("3-7-1-fish_length.csv")["length"]
fish


# ### 4. 实现：点估计

# In[3]:


# 总体均值的点估计
mu = sp.mean(fish)
mu


# In[4]:


# 总体方差的点估计
sigma_2 = sp.var(fish, ddof = 1)
sigma_2


# ### 9. 实现：区间估计

# In[5]:


# 自由度
df = len(fish) - 1
df


# In[6]:


# 标准误差
se = sigma / sp.sqrt(len(fish))
se


# In[8]:


# 区间估计
interval = stats.t.interval(
    alpha = 0.95, df = df, loc = mu, scale = se)
interval


# ### 10. 补充：置信区间的求解细节

# In[9]:


# 97.5% 分位数
t_975 = stats.t.ppf(q = 0.975, df = df)
t_975


# In[10]:


# 下置信界限
lower = mu - t_975 * se
lower


# In[11]:


# 上置信界限
upper = mu + t_975 * se
upper


# ### 11. 决定置信区间大小的因素

# In[12]:


# 样本方差越大, 置信区间越大
se2 = (sigma*10) / sp.sqrt(len(fish))
stats.t.interval(
    alpha = 0.95, df = df, loc = mu, scale = se2)


# In[13]:


# 样本容量越大, 置信区间越小
df2 = (len(fish)*10) - 1
se3 = sigma / sp.sqrt(len(fish)*10)
stats.t.interval(
    alpha = 0.95, df = df2, loc = mu, scale = se3)


# In[14]:


# 99% 置信区间
stats.t.interval(
    alpha = 0.99, df = df, loc = mu, scale = se)


# ### 12. 区间估计结果的解读

# In[19]:


# 如果置信区间包含总体均值 (4) 就取 True
be_included_array = np.zeros(20000, dtype = "bool")
be_included_array


# In[20]:


# 执行 20,000 次求 95% 置信区间的操作
# 如果置信区间包含总体均值 (4) 就取 True
np.random.seed(1)
norm_dist = stats.norm(loc = 4, scale = 0.8)
for i in range(0, 20000):
    sample = norm_dist.rvs(size = 10)
    df = len(sample) - 1
    mu = sp.mean(sample)
    std = sp.std(sample, ddof = 1)
    se = std / sp.sqrt(len(sample))
    interval = stats.t.interval(0.95, df, mu, se)
    if(interval[0] <= 4 and interval[1] >= 4):
        be_included_array[i] = True


# In[21]:


sum(be_included_array) / len(be_included_array)


# In[ ]:




