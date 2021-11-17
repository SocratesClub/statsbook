#!/usr/bin/env python
# coding: utf-8

# # 第 9 节　均值差的检验
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学
# 
# 

# ### 3. 实现：实验准备

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
paired_test_data = pd.read_csv(
    "3-9-1-paired-t-test.csv")
print(paired_test_data)


# ### 4. 实现：配对样本 t 检验

# In[3]:


# 服药前后的样本均值
before = paired_test_data.query(
    'medicine == "before"')["body_temperature"]
after = paired_test_data.query(
    'medicine == "after"')["body_temperature"]
# 转为数组类型
before = np.array(before)
after = np.array(after)
# 计算差值
diff = after - before
diff


# In[4]:


# 检验均值是否与 0 存在差异
stats.ttest_1samp(diff, 0)


# In[5]:


# 配对样本 t 检验
stats.ttest_rel(after, before)


# ### 6. 实现：独立样本 t 检验

# In[6]:


# 均值
mean_bef = sp.mean(before)
mean_aft = sp.mean(after)

# 方差
sigma_bef = sp.var(before, ddof = 1)
sigma_aft = sp.var(after, ddof = 1)

# 样本容量
m = len(before)
n = len(after)

# t 值
t_value = (mean_aft - mean_bef) /     sp.sqrt((sigma_bef/m + sigma_aft/n))
t_value


# In[7]:


stats.ttest_ind(after, before, equal_var = False)


# In[ ]:




