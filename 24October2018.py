
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


# In[23]:

import random


# In[40]:

random_a=np.random.randint(20,size=100)


# In[42]:

print(random_a.sorted())


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




# In[1]:

import numpy as np
x=np.arange(10)
x


# In[33]:

myL=createArrayVersion(test_sayisi)
myL+1


# In[2]:

import matplotlib.pyplot as plt


# In[98]:

image_1=plt.imread("image_1.jpg")
plt.imshow(image_1)
plt.show()


# In[5]:

image_1.shape


# In[8]:

type(image_1)


# In[10]:

image_1.ndim


# In[15]:

image_1.xdata


# In[14]:

image1.nparray


# In[13]:

image_1.nparray


# In[9]:

def my_function_1(myimage):
    print("Resimin Boyutu : ",myimage.ndim)
    print("\nResimin Çözünürlüğü : ",myimage.shape)
    print("\nR için min değer:",myimage[:,:,0].min(),",max değer:",myimage[:,:,0].max())
    print("\nG için min değer:",myimage[:,:,1].min(),",max değer:",myimage[:,:,1].max())
    print("\nB için min değer:",myimage[:,:,2].min(),",max değer:",myimage[:,:,2].max())


# In[10]:

my_function_1(image_1)


# In[20]:

image_2=plt.imread("image_2.jpg")


# In[19]:

my_function_1(image_2)


# In[20]:

image_3=plt.imread("image_3.jpg")


# In[21]:

my_function_1(image_2)


# In[24]:

image_1[:,:,1]=image_1[:,:,1]-100


# In[25]:

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


# In[13]:

import numpy as np
from scipy import stats
import operator


# In[11]:

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


# In[14]:

inverse(image_1)
mean(image_1)
mode(image_1)
median(image_1)
plt.imshow(image_1)
plt.show()
my_H(image_1)


# In[126]:

plt.imshow(image_1)
plt.show()


# In[6]:

def get_distance(v,w=[1/3,1/3,1/3]):
    a,b,c=v[0],v[1],v[2]
    w0,w1,w2=w[0],w[1],w[2]
    d=((a**2)*w0+(b**2)*w1+(c**2)*w2)**.5
    return d


# In[129]:

my_RGB=[10,20,3]
gray_level=get_distance(my_RGB,[.6,.3,.1])
print(gray_level)


# In[7]:

def convert_rgb_to_gray_level(im_1):
    im_2=np.zeros((im_1.shape[0],im_1.shape[1]))
    for i in range(im_1.shape[0]):
        for j in range(im_1.shape[1]):
            im_2[i,j]=get_distance(im_1[i,j,:])
    return im_2


# In[99]:

image_2=convert_rgb_to_gray_level(image_1)
plt.imshow(image_2, cmap='gray')
plt.show()


# In[95]:

def convert_gray_level_to_BW(image_gray_level):
    im_BW=np.zeros((image_gray_level.shape[0],image_gray_level.shape[1]))
    for i in range(image_gray_level.shape[0]):
        for j in range(image_gray_level.shape[1]):
            if(image_gray_level[i,j]>120):
                im_BW[i,j]=0
            else:
                im_BW[i,j]=1
    return im_BW


# In[100]:

im_3=convert_gray_level_to_BW(image_2)


# In[96]:

plt.imshow(image_1), plt.imshow(image_2, cmap='gray') ,plt.imshow(im_3, cmap='gray')


# In[97]:

plt.imshow(image_1)


# In[98]:

plt.imshow(image_2, cmap='gray')


# In[102]:

plt.imshow(im_3, cmap='binary')


# In[104]:

my_H(image_1)


# In[131]:

def my_H_for_Gray(image):
    H={}
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            if(image[i,j] in H.keys()):
                H[image[i,j]]=H[image[i,j]]+1
            else:
                H[image[i,j]]=1
    plt.title("İmage Histogrami")
    plt.bar(list(H.keys()), H.values(), color='r')
    plt.show()
def inverse_for_Gray(inv_image):
    for i in range(inv_image.shape[0]):
        for j in range(inv_image.shape[1]):
            inv_image[i,j]=255-inv_image[i,j]
    return inv_image
    


# In[140]:

my_H_for_Gray(image_2)


# In[147]:

plt.imshow(image_2, cmap='gray')


# In[148]:

image_2=inverse_for_Gray(image_2)


# In[142]:

my_H_for_Gray(image_2)


# In[149]:

plt.imshow(image_2, cmap='gray')


# In[63]:

mask_0=np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3)/9
mask_1=np.random.randint(5,size=9).reshape(3,3)
mask_2=np.random.randint(5,size=9).reshape(3,3)


# In[66]:

print(mask_0)
print("*********")
print(mask_1)
print("*********")
print(mask_2)
print("*********")
print(mask_1*mask_2)
print("*********")
print(mask_1*mask_0)


# In[70]:

sum(sum(mask_1*mask_0))


# In[89]:

def def_default_mask_for_mean():
    return np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3)/9
def apply_mask(part_of_image):
    mask=def_default_mask_for_mean()
    return sum(sum(part_of_image*mask))


# In[142]:

image_gray=convert_rgb_to_gray_level(image_1)
plt.imshow(image_gray, cmap="gray")
plt.show()


# In[145]:

m=image_gray.shape[0]
n=image_gray.shape[1]
image_gray_new=np.zeros((m,n))
for i in range(1,m-1):
    for j in range(1,n-1):
        poi=image_gray[i-1:i+2,j-1:j+2]
        image_gray_new[i,j]=apply_mask(poi)
    


# In[148]:

plt.imshow(image_gray, cmap="gray")
plt.show()


# In[149]:

plt.imshow(image_gray_new, cmap="gray")
plt.show()


# In[73]:

mask_1=np.random.randint(20,size=9).reshape(3,3)
mask_1


# In[139]:

plt.imshow(im_3, cmap="gray")
plt.show()
plt.imshow(image_2, cmap="gray")
plt.show()


# In[162]:

def get_median(part_of_image):
    part=part_of_image.reshape(1,9)
    part.sort()
    return part[0,4]


# In[163]:

m=image_gray.shape[0]
n=image_gray.shape[1]
image_gray_new2=np.zeros((m,n))
for i in range(1,m-1):
    for j in range(1,n-1):
        poi=image_gray[i-1:i+2,j-1:j+2]
        image_gray_new2[i,j]=get_median(poi)
    


# In[165]:

plt.imshow(image_gray_new2, cmap="gray")
plt.show()


# In[20]:

image_1.shape


# In[19]:

im_3.shape


# In[18]:

im_3[55,20]


# In[162]:

def pixel_compare_1(pixelx,pixely):
    temper=[]
    temper.append(image_3[pixelx-1,pixely-1]) 
    temper.append(image_3[pixelx-1,pixely]) 
    temper.append(image_3[pixelx,pixely-1]) 
    temper.append(image_3[pixelx,pixely]) 
    if(temper==[0,0,0,1] or temper==[0,0,1,0] or temper==[0,1,0,0] or temper==[1,0,0,0]):
        return True
    else:
        return False
def pixel_compare_2(pixelx,pixely):
    temper=[]
    temper.append(image_3[pixelx-1,pixely-1]) 
    temper.append(image_3[pixelx-1,pixely]) 
    temper.append(image_3[pixelx,pixely-1]) 
    temper.append(image_3[pixelx,pixely])         
    if(temper==[1,1,1,0] or temper==[1,1,0,1] or temper==[1,0,1,1] or temper==[0,1,1,1]):
        return True
    else:
        return False
def object_count(imagetest):
    m=imagetest.shape[0]
    n=imagetest.shape[1]
    e=0
    i=0
    for j in range(1,m-1):
        for k in range(i,n-1):
            if(pixel_compare_1(j,k)==True):
                e=e+1
            elif(pixel_compare_2(j,k)==True):
                i=i+1
    print(e,i)
    print((e-i)/4)


# In[163]:

def test():
    if([1,0,1,2]==[1.0,0.0,1.0,1.0] or [1,0,1,2]==[1.0,0.0,1.0,2.0]):
        print("true")


# In[194]:

image_1=plt.imread("image_1.jpg")
image_2=convert_rgb_to_gray_level(image_1)
image_3=convert_gray_level_to_BW(image_2)
object_count(image_3)


# In[195]:

plt.imshow(image_3, cmap='binary')
plt.show()


# In[ ]:



