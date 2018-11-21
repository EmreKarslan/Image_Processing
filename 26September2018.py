
# coding: utf-8

# In[1]:

test_0="DIP= Digital Image Processing"
test_1="Goruntu Isleme"


# In[2]:

print(test_0)


# In[3]:

var_0=10
var_1=100


# In[4]:

var_1


# In[5]:

var_1+var_0


# In[6]:

myList=[0,1,"2"]
print(myList)


# In[7]:

myList


# In[8]:

myList[1]
myList[2]


# In[9]:

myList[1]


# In[10]:

myList.count


# In[11]:

for i in myList:
    print(str(i)+"1")


# In[12]:

for i in myList:
    print(int(i)+1)


# In[13]:

myList_0


# In[14]:

for i in myList:
    myList_0.append(int(i)+1)


# In[15]:

myList_0=[]


# In[16]:

for i in myList:
    myList_0.append(int(i)+1)


# In[17]:

print(len(myList_0))


# In[18]:

import random


# In[19]:

random_a=random.randint(1,100)


# In[20]:

random_a


# In[21]:

myList_1=[]
for i in range(10):
    myList_1.append(random.randint(1,100))
myList_1


# In[22]:

def createArray(s):
    myList=[]
    for i in range(s):
        myList.append(random.randint(1,10))
    return myList
def createArrayVersion(s):
    myList=np.arange(s)
    return myList
def printArray(array):
    print(array)
def incArray(array):
    myList_1=[]
    for i in array:
        myList_1.append(i+1)
    return myList_1


# In[23]:

test_sayisi=1000000


# In[ ]:




# In[32]:

import numpy as np
x=np.arange(10)
x


# In[33]:

myL=createArrayVersion(test_sayisi)
myL+1


# In[34]:

import matplotlib.pyplot as plt


# In[85]:

image_1=plt.imread("image_1.jpg")
plt.imshow(image_1)
plt.show()


# In[30]:

image_1.shape


# In[31]:

type(image_1)


# In[35]:

image_1.ndim


# In[36]:

image_1.xdata


# In[37]:

image1.nparray


# In[38]:

image_1.nparray


# In[56]:

def my_function_1(myimage):
    print("Resimin Boyutu : ",myimage.ndim)
    print("\nResimin Çözünürlüğü : ",myimage.shape)
    print("\nR için min değer:",myimage[:,:,0].min(),",max değer:",myimage[:,:,0].max())
    print("\nG için min değer:",myimage[:,:,1].min(),",max değer:",myimage[:,:,1].max())
    print("\nB için min değer:",myimage[:,:,2].min(),",max değer:",myimage[:,:,2].max())


# In[97]:

my_function_1(image_1)


# In[62]:

image_2=plt.imread("image_2.jpg")


# In[59]:

my_function_1(image_2)


# In[61]:

image_3=plt.imread("image_3.jpg")


# In[63]:

my_function_1(image_3)


# In[135]:

image_1[:,:,0]=image_1[:,:,0]-100


# In[144]:

plt.imshow(image_1)
plt.show()


# In[145]:

image_1[:,:,0]=image_1[:,:,0]+100


# In[148]:

for i in range(768):
    for j in range(1024):
        if(image_1[i,j,0]==0):
            count=count+1


# In[147]:

count=0


# In[133]:

count


# In[139]:

count


# In[143]:

count


# In[149]:

count


# In[ ]:



