#!/usr/bin/env python
# coding: utf-8

# # 第 2 节　使用 Python 进行描述统计：多变量
# 
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学
# 
# 

# ### 4. 多变量数据的管理

# In[1]:


# 用于数值计算的库
import pandas as pd
import scipy as sp

# 设置浮点数打印精度
get_ipython().run_line_magic('precision', '3')


# ### 5. 实现：求各分组的统计量

# In[2]:


fish_multi = pd.read_csv("3-2-1-fish_multi.csv")
print(fish_multi)


# In[3]:


# 按鱼的种类计算
group = fish_multi.groupby("species")
print(group.mean())


# In[4]:


print(group.std(ddof = 1))


# In[5]:


group.describe()


# ### 6. 实现：列联表

# In[6]:


shoes = pd.read_csv("3-2-2-shoes.csv")
print(shoes)


# In[7]:


cross = pd.pivot_table(
    data = shoes,
    values = "sales",
    aggfunc = "sum",
    index = "store",
    columns = "color"
)
print(cross)


# ### 9. 实现：协方差

# In[8]:


cov_data = pd.read_csv("3-2-3-cov.csv")
print(cov_data)


# In[9]:


# 读取数据的列
x = cov_data["x"]
y = cov_data["y"]

# 求样本容量
N = len(cov_data)

# 求各变量均值
mu_x = sp.mean(x)
mu_y = sp.mean(y)


# In[10]:


# 样本协方差
cov_sample = sum((x - mu_x) * (y - mu_y)) / N
cov_sample


# In[11]:


# 协方差
cov = sum((x - mu_x) * (y - mu_y)) / (N - 1)
cov


# ### 10. 实现：协方差矩阵

# In[12]:


# 样本协方差
sp.cov(x, y, ddof = 0)


# In[13]:


# 无偏协方差
sp.cov(x, y, ddof = 1)


# ### 13. 实现：皮尔逊积矩相关系数

# In[14]:


# 计算两个变量的方差
sigma_2_x = sp.var(x, ddof = 1)
sigma_2_y = sp.var(y, ddof = 1)

# 计算相关系数
rho = cov / sp.sqrt(sigma_2_x * sigma_2_y)
rho


# In[15]:


# 计算两个变量的方差
sigma_2_x_sample = sp.var(x, ddof = 0)
sigma_2_y_sample = sp.var(y, ddof = 0)

# 计算相关系数
cov_sample / sp.sqrt(sigma_2_x_sample * sigma_2_y_sample)


# In[16]:


# 相关矩阵
sp.corrcoef(x, y)


# In[ ]:




