#!/usr/bin/env python
# coding: utf-8

# # 第 10 节　列联表检验
# 
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学

# ### 5. 实现：计算 p 值

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


# 计算 p 值
1 - sp.stats.chi2.cdf(x = 6.667, df = 1)


# ### 6. 实现：列联表检验

# In[3]:


# 读入数据
click_data = pd.read_csv("3-10-1-click_data.csv")
print(click_data)


# In[4]:


# 转换为列联表
cross = pd.pivot_table(
    data = click_data,
    values = "freq",
    aggfunc = "sum",
    index = "color",
    columns = "click"
)
print(cross)


# In[5]:


# 进行检验
sp.stats.chi2_contingency(cross, correction = False)

