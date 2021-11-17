#!/usr/bin/env python
# coding: utf-8

# # 第 3 节　基于 matplotlib、seaborn 的数据可视化
# 
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学

# ### 2. 实现：数据可视化的环境准备

# In[1]:


# 用于数值计算的库
import numpy as np
import pandas as pd

# 设置浮点数打印精度
get_ipython().run_line_magic('precision', '3')

# 用于绘图的库
from matplotlib import pyplot as plt

# 在 Jupyter Notebook 里显示图形
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 3. 实现：用 pyplot 绘制折线图

# In[2]:


x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([2,3,4,3,5,4,6,7,4,8])


# In[3]:


plt.plot(x, y, color = 'black')
plt.title("lineplot matplotlib")
plt.xlabel("x")
plt.ylabel("y")


# ### 4. 实现：用 seaborn 和 pyplot 绘制折线图

# In[4]:


import seaborn as sns
sns.set()


# In[5]:


plt.plot(x, y, color = 'black')
plt.title("lineplot seaborn")
plt.xlabel("x")
plt.ylabel("y")


# ### 5. 实现：用 seaborn 绘制直方图

# In[6]:


fish_data = np.array([2,3,3,4,4,4,4,5,5,6])
fish_data


# In[7]:


sns.distplot(fish_data, bins = 5, 
             color = 'black', kde = False)


# ### 6. 实现：通过核密度估计将直方图平滑化

# In[8]:


sns.distplot(fish_data, bins = 1, 
             color = 'black', kde = False)


# In[9]:


sns.distplot(fish_data, color = 'black')


# ### 7. 实现：两个变量的直方图

# In[10]:


fish_multi = pd.read_csv("3-3-2-fish_multi_2.csv")
print(fish_multi)


# In[11]:


print(fish_multi.groupby("species").describe())


# In[12]:


# 按鱼的种类区分数据
length_a = fish_multi.query('species == "A"')["length"]
length_b = fish_multi.query('species == "B"')["length"]


# In[13]:


# 绘制这两个直方图
sns.distplot(length_a, bins = 5, 
             color = 'black', kde = False)
sns.distplot(length_b, bins = 5, 
             color = 'gray', kde = False)


# ### 9. 实现：箱形图

# In[14]:


# 箱形图
sns.boxplot(x = "species", y  = "length", 
            data = fish_multi, color = 'gray')


# In[15]:


fish_multi.groupby("species").describe()


# ### 10. 实现：小提琴图

# In[16]:


sns.violinplot(x = "species", y  = "length", 
               data = fish_multi, color = 'gray')


# ### 11. 实现：条形图

# In[17]:


sns.barplot(x = "species", y  = "length", 
            data = fish_multi, color = 'gray')


# ### 12. 实现：散点图

# In[18]:


cov_data = pd.read_csv("3-2-3-cov.csv")
print(cov_data)


# In[19]:


sns.jointplot(x = "x", y = "y", 
              data = cov_data, color = 'black')


# ### 13. 实现：散点图矩阵

# In[20]:


# 导入 seaborn 内置的鸢尾花数据
iris = sns.load_dataset("iris")
iris.head(n = 3)


# In[21]:


# 每种类鸢尾花各个规格的均值
iris.groupby("species").mean()


# In[22]:


# 散点图矩阵
sns.pairplot(iris, hue="species", palette='gray')


# In[ ]:




