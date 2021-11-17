#!/usr/bin/env python
# coding: utf-8

# # 第 5 节　样本统计量的性质
# 
# ## 第 3 章　使用 Pyhton 进行数据分析｜用 Python 动手学统计学
# 
# 

# ### 3. 导入所需的库

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


# 总体服从均值为 4 标准差为 0.8 的正态分布
population = stats.norm(loc = 4, scale = 0.8)


# ### 4. 多次计算样本均值

# In[3]:


# 存放均值的容器
sample_mean_array = np.zeros(10000)


# In[4]:


# 抽取 10 个数据并计算均值, 此操作重复 10,000 次
np.random.seed(1)
for i in range(0, 10000):
    sample = population.rvs(size = 10)
    sample_mean_array[i] = sp.mean(sample)


# In[5]:


sample_mean_array


# ### 5. 样本均值的均值与总体均值相近

# In[6]:


# 样本均值的均值
sp.mean(sample_mean_array)


# In[7]:


# 样本均值的标准差
sp.std(sample_mean_array, ddof = 1)


# In[8]:


# 样本均值的样本分布
sns.distplot(sample_mean_array, color = 'black')


# ### 6. 样本容量越大，样本均值越接近总体均值

# In[9]:


# 公差是 100 的样本容量, 范围是 10 到 100,010
size_array =  np.arange(
    start = 10, stop = 100100, step = 100)
size_array


# In[10]:


# 存放样本均值的容器
sample_mean_array_size = np.zeros(len(size_array))


# In[11]:


# 改变样本容量的同时计算样本均值
np.random.seed(1)
for i in range(0, len(size_array)):
    sample = population.rvs(size = size_array[i])
    sample_mean_array_size[i] = sp.mean(sample)


# In[12]:


plt.plot(size_array, sample_mean_array_size, 
         color = 'black')
plt.xlabel("sample size")
plt.ylabel("sample mean")


# ### 7. 定义用于计算样本均值的函数

# In[13]:


# 用于计算样本均值的函数
def calc_sample_mean(size, n_trial):
    sample_mean_array = np.zeros(n_trial)
    for i in range(0, n_trial):
        sample = population.rvs(size = size)
        sample_mean_array[i] = sp.mean(sample)
    return(sample_mean_array)


# In[14]:


# 验证函数功能
# 抽取 10 个数据并计算均值, 此操作重复 10,000 次, 再求这 10,000 个结果的均值
np.random.seed(1)
sp.mean(calc_sample_mean(size = 10, n_trial = 10000))


# ### 8. 不同样本容量所得的样本均值的分布

# In[15]:


np.random.seed(1)
# 样本容量 10
size_10 = calc_sample_mean(size = 10, n_trial = 10000)
size_10_df = pd.DataFrame({
    "sample_mean":size_10,
    "size"       :np.tile("size 10", 10000)
})
# 样本容量 20
size_20 = calc_sample_mean(size = 20, n_trial = 10000)
size_20_df = pd.DataFrame({
    "sample_mean":size_20,
    "size"       :np.tile("size 20", 10000)
})
# 样本容量 30
size_30 = calc_sample_mean(size = 30, n_trial = 10000)
size_30_df = pd.DataFrame({
    "sample_mean":size_30,
    "size"       :np.tile("size 30", 10000)
})

# 拼接表格
sim_result = pd.concat(
    [size_10_df, size_20_df, size_30_df])

# 打印结果
print(sim_result.head())


# In[16]:


sns.violinplot(x = "size", y = "sample_mean", 
               data = sim_result, color = 'gray')


# ### 9. 样本均值的标准差小于总体标准差

# In[17]:


# 公差为 2 的样本容量, 范围是 2 到 100
size_array =  np.arange(
    start = 2, stop = 102, step = 2)
size_array


# In[18]:


# 存放样本均值的标准差的容器
sample_mean_std_array = np.zeros(len(size_array))


# In[19]:


# 改变样本容量的同时计算样本均值的标准差
np.random.seed(1)
for i in range(0, len(size_array)):
    sample_mean = calc_sample_mean(size =size_array[i], 
                                   n_trial = 100)
    sample_mean_std_array[i] = sp.std(sample_mean, 
                                      ddof = 1)


# In[20]:


plt.plot(size_array, sample_mean_std_array, 
         color = 'black')
plt.xlabel("sample size")
plt.ylabel("mean_std value")


# ### 10. 标准误差

# In[21]:


# 样本均值的标准差的理论值：标准误差
standard_error = 0.8 / np.sqrt(size_array)
standard_error


# In[22]:


plt.plot(size_array, sample_mean_std_array, 
         color = 'black')
plt.plot(size_array, standard_error, 
         color = 'black', linestyle = 'dotted')
plt.xlabel("sample size")
plt.ylabel("mean_std value")


# ### 12. 样本方差的均值偏离于总体方差

# In[23]:


# 存放方差值的容器
sample_var_array = np.zeros(10000)


# In[24]:


# 取出 10 个数据并求其方差, 执行 10,000 次
np.random.seed(1)
for i in range(0, 10000):
    sample = population.rvs(size = 10)
    sample_var_array[i] = sp.var(sample, ddof = 0)


# In[25]:


# 样本方差的均值
sp.mean(sample_var_array)


# ### 13. 采用无偏方差消除偏离

# In[26]:


# 存放无偏方差的空间
unbias_var_array = np.zeros(10000)
# 进行 10,000 次计算10个数据的无偏方差操作
# 
np.random.seed(1)
for i in range(0, 10000):
    sample = population.rvs(size = 10)
    unbias_var_array[i] = sp.var(sample, ddof = 1)
# 无偏方差的均值
sp.mean(unbias_var_array)


# ### 14. 样本容量越大，其无偏方差越接近总体方差

# In[27]:


# 公差为 100 的样本容量, 范围是 10 到 100,010
size_array =  np.arange(
    start = 10, stop = 100100, step = 100)
size_array


# In[28]:


# 存放无偏方差的容器
unbias_var_array_size = np.zeros(len(size_array))


# In[29]:


# 在样本容量变化的同时反复计算样本的无偏方差
np.random.seed(1)
for i in range(0, len(size_array)):
    sample = population.rvs(size = size_array[i])
    unbias_var_array_size[i] = sp.var(sample, ddof = 1)


# In[30]:


plt.plot(size_array, unbias_var_array_size, 
         color = 'black')
plt.xlabel("sample size")
plt.ylabel("unbias var")


# ### 19. 补充：中心极限定理

# In[31]:


# 样本容量与试验次数
n_size  = 10000
n_trial = 50000
# 正面为 1, 背面为 0
coin = np.array([0,1])
# 出现正面的次数
count_coin = np.zeros(n_trial)
# 投 n_size 次硬币, 此实验进行 n_trial 次
np.random.seed(1)
for i in range(0, n_trial):
    count_coin[i] = sp.sum(
        np.random.choice(coin, size = n_size, 
                         replace = True))
# 绘出直方图
sns.distplot(count_coin, color = 'black')


# In[ ]:




