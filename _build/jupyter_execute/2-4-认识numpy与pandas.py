#!/usr/bin/env python
# coding: utf-8

# # 第 4 节　认识 numpy 与 pandas
# 
# ## 第 2 章　Python 与 Jupyter Notebook 基础｜用 Python 动手学统计学

# ### 1. 导入用于分析的功能

# In[1]:


import numpy as np
import pandas as pd


# ### 3. 实现：列表

# In[2]:


sample_list = [1,2,3,4,5]
sample_list


# ### 5. 实现：数组

# In[3]:


sample_array = np.array([1,2,3,4,5])
sample_array


# In[4]:


sample_array + 2


# In[5]:


sample_array * 2


# In[6]:


np.array([1 ,2, "A"])


# In[7]:


# 矩阵
sample_array_2 = np.array(
    [[1,2,3,4,5],
     [6,7,8,9,10]])
sample_array_2


# In[8]:


# 获取行数与列数
sample_array_2.shape


# ### 6. 实现：生成等差数列的方法

# In[9]:


np.arange(start = 1, stop = 6, step = 1)


# In[10]:


np.arange(start = 0.1, stop = 0.8, step = 0.2)


# In[11]:


np.arange(0.1, 0.8, 0.2)


# ### 7. 实现：多种生成数组的方式

# In[12]:


# 元素相同的数组
np.tile("A", 5)


# In[13]:


# 存放 4 个 0
np.tile(0, 4)


# In[14]:


# 只有 0 的数组
np.zeros(4)


# In[15]:


# 二维数组
np.zeros([2,3])


# In[16]:


# 只有 1 的数组
np.ones(3)


# ### 8. 实现：切片

# In[17]:


# 一维数组
d1_array = np.array([1,2,3,4,5])
d1_array


# In[18]:


# 取得第一个元素
d1_array[0]


# In[19]:


# 获取索引中的 1 号和 2 号元素
d1_array[1:3]


# In[20]:


# 二维数组
d2_array = np.array(
    [[1,2,3,4,5],
    [6,7,8,9,10]])
d2_array


# In[21]:


d2_array[0, 3]


# In[22]:


d2_array[1, 2:4]


# ### 9. 实现：数据帧

# In[23]:


sample_df = pd.DataFrame({
    'col1' : sample_array, 
    'col2' : sample_array * 2,
    'col3' : ["A", "B", "C", "D", "E"]
})
print(sample_df)


# In[24]:


sample_df


# ### 10. 实现：读取文件中的数据

# In[25]:


file_data = pd.read_csv("2-4-1-sample_data.csv")
print(file_data)


# In[26]:


type(file_data)


# ### 11. 实现：连接数据帧

# In[27]:


df_1 = pd.DataFrame({
    'col1' : np.array([1, 2, 3]),
    'col2' : np.array(["A", "B", "C"])
})
df_2 = pd.DataFrame({
    'col1' : np.array([4, 5, 6]),
    'col2' : np.array(["D", "E", "F"])
})


# In[28]:


# 在纵向上连接
print(pd.concat([df_1, df_2]))


# In[29]:


# 在横向上连接
print(pd.concat([df_1, df_2], axis = 1))


# ### 12. 实现：数据帧的列操作

# In[30]:


# 对象数据
print(sample_df)


# In[31]:


# 按列名获取数据
print(sample_df.col2)


# In[32]:


print(sample_df["col2"])


# In[33]:


print(sample_df[["col2", "col3"]])


# In[34]:


# 删除指定的列
print(sample_df.drop("col1", axis = 1))


# ### 13. 实现：数据帧的行操作

# In[35]:


# 获取前 3 行
print(sample_df.head(n = 3))


# In[36]:


# 获取第 1 行
print(sample_df.query('index == 0'))


# In[37]:


# 通过多种条件获取数据
print(sample_df.query('col3 == "A"'))


# In[38]:


# 按 OR 条件获取数据
print(sample_df.query('col3 == "A" | col3 == "D"'))


# In[39]:


# 按 AND 条件获取数据
print(sample_df.query('col3 == "A" & col1 == 3'))


# In[40]:


# 同时指定行和列的条件
print(sample_df.query('col3 == "A"')[["col2", "col3"]])


# ### 14. 补充：序列

# In[41]:


type(sample_df)


# In[42]:


type(sample_df.col1)


# In[43]:


# 转换为数组
type(np.array(sample_df.col1))


# In[44]:


type(sample_df.col1.values)


# ### 15. 补充：函数文档

# In[45]:


help(sample_df.query)


# In[ ]:




