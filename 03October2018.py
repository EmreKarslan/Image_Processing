
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




# In[2]:

import numpy as np
x=np.arange(10)
x


# In[33]:

myL=createArrayVersion(test_sayisi)
myL+1


# In[4]:

import matplotlib.pyplot as plt


# In[5]:

image_1=plt.imread("image_1.jpg")
plt.imshow(image_1)
plt.show()


# In[8]:

image_1.shape


# In[31]:

type(image_1)


# In[10]:

image_1.ndim


# In[9]:

image_1.xdata


# In[37]:

image1.nparray


# In[38]:

image_1.nparray


# In[11]:

def my_function_1(myimage):
    print("Resimin Boyutu : ",myimage.ndim)
    print("\nResimin Çözünürlüğü : ",myimage.shape)
    print("\nR için min değer:",myimage[:,:,0].min(),",max değer:",myimage[:,:,0].max())
    print("\nG için min değer:",myimage[:,:,1].min(),",max değer:",myimage[:,:,1].max())
    print("\nB için min değer:",myimage[:,:,2].min(),",max değer:",myimage[:,:,2].max())


# In[12]:

my_function_1(image_1)


# In[51]:

image_2=plt.imread("image_2.jpg")


# In[52]:

my_function_1(image_2)


# In[19]:

image_3=plt.imread("image_3.jpg")


# In[41]:

my_function_1(image_2)


# In[21]:

image_1[:,:,1]=image_1[:,:,1]+100


# In[24]:

plt.imshow(image_1)
plt.show()


# In[39]:

image_2[:,:,:]=image_2[:,:,:]-200


# In[53]:

plt.imshow(image_2)
plt.show()


# In[49]:

for i in range(768):
    for j in range(1024):
        for k in range(3):
            image_2[i,j,k]=0+image_2[i,j,k]


# In[58]:

for i in range(3):
    for k in range(255):
        myRGB[i,k]=0


# In[96]:

import numpy
H_0= [0 for i in range (256)]
H_1= [0 for i in range (256)]
H_2= [0 for i in range (256)]


# for i in range(25):
#     for j in range(25):
#         for k in range(3):
#             if(k==0):
#                 H_1[0,image_2[i,j,k]]=H_1[0,image_2[i,j,k]]+1
#             elif(k==1):
#                 H_1[1,image_2[i,j,k]]=H_1[1,image_2[i,j,k]]+1
#             elif(k==2):
#                 H_1[2,image_2[i,j,k]]=H_1[2,image_2[i,j,k]]+1    

# In[100]:

for i in range(image_2.shape[0]): 
    for j in range(image_2.shape[1]): 
        for k in range(3): 
            if(k==0): 
                H_0[image_2[i,j,0]]=H_0[image_2[i,j,0]]+1 
            elif(k==1): 
                H_1[image_2[i,j,1]]=H_1[image_2[i,j,1]]+1 
            elif(k==2): 
                H_2[image_2[i,j,2]]=H_2[image_2[i,j,2]]+1


# In[ ]:

def my_H2(image):
    H={}
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            for k in range(3): 
                if(k==0): 
                    H_0[image_2[i,j,0]]=H_0[image_2[i,j,0]]+1 
                elif(k==1): 
                    H_1[image_2[i,j,1]]=H_1[image_2[i,j,1]]+1 
                elif(k==2): 
                    H_2[image_2[i,j,2]]=H_2[image_2[i,j,2]]+1
    H=[]
    H.append(H_0)
    H.append(H_1)
    H.append(H_2)
    return H


# In[26]:

import numpy as np
from scipy import stats
import operator


# In[34]:

def my_function(x):
    return 255-x

def inverse(image):
    image[:,:,0]=my_function(image[:,:,0])
    image[:,:,1]=my_function(image[:,:,1])
    image[:,:,2]=my_function(image[:,:,2])
    
def mean(image):
    print("Kirmizi icin renk ortalamasi : ",np.mean(image[:,:,0]))
    print("Yesil icin renk ortalamasi : ",np.mean(image[:,:,1]))
    print("Mavi icin renk ortalamasi : ",np.mean(image[:,:,2]))
    
def median(image):
    print("Kirmizi icin renk ortalamasi : ",np.median(image[:,:,0]))
    print("Yesil icin renk ortalamasi : ",np.median(image[:,:,1]))
    print("Mavi icin renk ortalamasi : ",np.median(image[:,:,2]))
    
def mode(image):
    print("Kirmizi icin renk ortalamasi : ",stats.mode(image[:,:,0]))
    print("Yesil icin renk ortalamasi : ",stats.mode(image[:,:,1]))
    print("Mavi icin renk ortalamasi : ",stats.mode(image[:,:,2]))
    
def my_H(image):
    H={}
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            if(image[i,j,0] in H.keys()):
                H[image[i,j,0]]=H[image[i,j,0]]+1
            else:
                H[image[i,j,0]]=1
            if(image[i,j,1] in H.keys()):
                H[image[i,j,1]]=H[image[i,j,1]]+1
            else:
                H[image[i,j,1]]=1
            if(image[i,j,2] in H.keys()):
                H[image[i,j,2]]=H[image[i,j,2]]+1
            else:
                H[image[i,j,2]]=1
    plt.title("İmage Histogrami")
    plt.bar(list(H.keys()), H.values(), color='r')
    plt.show()
    sorted_x = sorted(H.items(), key=operator.itemgetter(1))
    print(sorted_x)
    H.values(){64}


# In[35]:

inverse(image_1)
mean(image_1)
mode(image_1)
median(image_1)
plt.imshow(image_1)
plt.show()
my_H(image_1)


# In[ ]:

plt.imshow(image_1)
plt.show()


# In[ ]:



