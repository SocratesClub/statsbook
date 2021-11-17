#!/usr/bin/env python
# coding: utf-8

# # 第 3 节　Python 编程基础
# 
# ##  第 2 章　Python 与 Jupyter Notebook 基础｜用 Python 动手学统计学

# ### 1. 实现：四则运算

# In[1]:


1 + 1


# In[2]:


5 - 2


# In[3]:


2 * 3


# In[4]:


2 ** 3


# In[5]:


6 / 3


# In[6]:


7 // 3


# ### 2. 实现：编写注释

# In[7]:


# 1 + 1


# ### 3. 实现：数据类型

# In[8]:


"A"


# In[9]:


'A'


# In[10]:


# 字符串
type("A")


# In[11]:


type('A')


# In[12]:


# 整型
type(1)


# In[13]:


# 浮点型
type(2.4)


# In[14]:


# 布尔型
type(True)


# In[15]:


# 布尔型
type(False)


# In[16]:


"A" + 1


# ### 4. 实现：比较运算符

# In[17]:


1 > 0.89


# In[18]:


3 >= 2


# In[19]:


3 < 2


# In[20]:


3 <= 2


# In[21]:


3 == 2


# In[22]:


3 != 2


# ### 5. 实现：变量

# In[23]:


x = 2
x + 1


# ### 6. 实现：函数

# In[24]:


(x + 2) * 4


# In[25]:


def sample_function(data):
    return((data + 2) * 4)


# In[26]:


sample_function(x)


# In[27]:


sample_function(3)


# In[28]:


sample_function(x) + sample_function(3)


# ### 7. 实现：类与实例

# In[29]:


class Sample_Class:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        
    def method2(self):
        return(self.data1 + self.data2)


# In[30]:


sample_instance = Sample_Class(data1 = 2, data2 = 3)


# In[31]:


sample_instance.data1


# In[32]:


sample_instance.method2()


# ### 8. 实现：基于 if 语句的程序分支

# In[3]:


data = 1
if(data < 2):
    print("数字小于 2")
else:
    print("数字不小于 2")


# In[2]:


data = 3
if(data < 2):
    print("数字小于 2")
else:
    print("数字不小于 2")


# ### 9. 实现：基于 for 语句的循环

# In[35]:


range(0, 3)


# In[36]:


for i in range(0, 3):
    print(i)


# In[37]:


for i in range(0, 3):
    print("hello")


# In[ ]:




