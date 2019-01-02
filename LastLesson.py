#!/usr/bin/env python
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


# In[16]:


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





# In[66]:


import numpy as np
x=np.arange(10)
x


# In[33]:


myL=createArrayVersion(test_sayisi)
myL+1


# In[3]:


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


# In[2]:


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


# In[9]:


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


# In[6]:


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


# In[11]:


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


# In[14]:


image_1=plt.imread("image_1.jpg")
image_2=convert_rgb_to_gray_level(image_1)
image_3=convert_gray_level_to_BW(image_2)
object_count(image_3)


# In[15]:


plt.imshow(image_3, cmap='binary')
plt.show()


# In[ ]:


#28x28 boyutlarından içeriği 0 ve 1 olan
#yukarıda üretilen matriste 1 leri içeren MBR(Diktörtgen üreten) fonksiyonu yazınız.
#Kendisine aktarılan 2 vektörün benzerliğini Return eden fonksiyonu Yazınız.
#En yukarıda yazdığını fonk kullanarak 100 farklı matris elde edip. 1'ci üretilen ile diğerlerini karşılaştırıp yakınlığını benzerliğini listeleyiniz.


# In[114]:





# In[115]:


random_a=random_a.reshape(28,28)


# In[105]:


random_a


# In[124]:


plt.imshow(random_a, cmap='binary')
plt.show()


# In[204]:


def create_matris(size1,size2):
    random_a=np.random.randint(2,size=(size1*size2))
    random_a=random_a.reshape(28,28)
    return random_a
def bounding_box(coords):
    m=coords.shape[0]
    n=coords.shape[1]
    x_min=m
    x_max=0
    y_min=n
    y_max=0
    a=coords.shape[0]
    b=coords.shape[1]
    for i in range (m):
        for j in range (n):
            if(coords[i,j]==1 and x_min>i):
                x_min=i
            if(coords[i,j]==1 and x_max<i):
                x_max=i
            if(coords[i,j]==1 and y_min>j):
                y_min=j
            if(coords[i,j]==1 and y_max<j):
                y_max=j
    return [(x_min,y_min),(x_max,y_min),(x_max,y_max),(x_min,y_max)]    
def get_similarity(character_a,character_b):
    m=character_a.shape[0]
    n=character_a.shape[1]
    my_similarity=0
    for i in range(m):
        for j in range(n):
            my_similarity=my_similarity+character_a[i,j]*character_b[i,j]
    return my_similarity
def create_100_matris_and_get_similarity():
    max_similarity_position=0
    max_similarity=0
    my_matris=[]
    my_similaritys=[]
    for i in range(100):
        my_matris.append(create_matris(28,28))
    for i in range(1,100):
        my_similaritys.append(get_similarity(my_matris[0],my_matris[i]))
        print("0 .... ",i," ",get_similarity(my_matris[0],my_matris[i]))
        if(max_similarity<get_similarity(my_matris[0],my_matris[i])):
            max_similarity=get_similarity(my_matris[0],my_matris[i])
            max_similarity_position=i
    return my_similaritys[max_similarity_position-1],max_similarity_position
    


# In[205]:


create_100_matris_and_get_similarity()


# In[103]:


def my_product_two_dim_with_threshold(a,b):
    return a[0]*b[0]+a[1]*b[2]*a[2]*b[2]
def get_my_data():
    my_data_x=[]
    my_data_x.append([1,0,0])
    my_data_x.append([1,0,1])
    my_data_x.append([1,1,0])
    my_data_x.append([1,1,1])
    my_data_x
    
    my_data_y=[]
    my_data_y.append(0)
    my_data_y.append(0)
    my_data_y.append(0)
    my_data_y.append(1)
    my_data_y
    
    return (my_data_x,my_data_y)


# In[8]:





# In[92]:


x,y=get_my_data()
for a,b in zip(x,y):
    print(a,b)


# In[99]:


def get_parameters():
    w=[]
    w.append(3)
    w.append(2)
    w.append(1)
    w
    learning_data=1
    epock=100
    
    return w,learning_data,epock
get_parameters()


# In[100]:


w,learning_rate,epoch=get_parameters()
samples,output=get_my_data()


# In[101]:


samples


# In[104]:


for i in range(epoch):
    error="Hata Yok"
    print("|*********************************|")
    for each_sample,d in zip(samples,output):
        print("Agirlik : ",w)
        print("Ornek : ",each_sample)
        print("Gercek Output : ",d)
        u=my_product_two_dim_with_threshold(each_smaple,w) #u=w*x
        #print(u)
        if(u>0):  #y=signal(u)
            y=1
        else:
            y=0
        print("Tahmini Cikti : ",y)
        if(y!=d): #error var
            for s in range(2):
                w[s]=w[s]-learning_rate*(y-d)*each_sample[s]
                error="Hata Var"
        print()
    if(error=="Hata Yok"):
        print("Hata Yok")
        break #return 0


# In[172]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
iris= datasets.load_iris()


# In[173]:


X_realdata = iris.data[:, :4]
X = iris.data[:, 2:4]  # we only take the first two features.
y = iris.target


# In[185]:


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
x1_mean,x2_mean=X[:,0].mean(),X[:,1].mean()
plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

fig = plt.figure(1, figsize=(8, 6))


ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=4).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 2], X_reduced[:, 3], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[34]:


x1_mean


# In[97]:


def iris_class_splice(a):
    iris_setosa=a[0:50,:]
    iris_versicolor=a[50:100,:]
    iris_verginica=a[100:150,:]
    return iris_setosa,iris_versicolor,iris_verginica


# In[151]:


def iris_class_mean_calc(data):
    Iris_means=[0,0,0,0]
    for i in range(data.__len__()):
            Iris_means[0]=Iris_means[0]+data[i,0]
            Iris_means[1]=Iris_means[1]+data[i,1]
            Iris_means[2]=Iris_means[2]+data[i,2]
            Iris_means[3]=Iris_means[3]+data[i,3]
    for j in range(4):
        Iris_means[j]=Iris_means[j]/50
    return Iris_means


# In[164]:


def iris_class_varyans_and_standart_sapma_calc(data,data_mean):
    Iris_varyans=np.zeros((4))
    Iris_standart_sapma=np.zeros((4))
    for j in range(data.__len__()):
        Iris_varyans[0]+=(data_mean[0]-data[j,0])**2
        Iris_varyans[1]+=(data_mean[1]-data[j,1])**2
        Iris_varyans[2]+=(data_mean[2]-data[j,2])**2
        Iris_varyans[3]+=(data_mean[3]-data[j,3])**2
    for j in range(4):
        Iris_varyans[j]=Iris_varyans[j]/50
        Iris_standart_sapma[j]=(Iris_varyans[j]/50)**0.5
    return Iris_varyans,Iris_standart_sapma
        


# In[165]:


a,b,c=iris_class_splice(X_realdata)


# In[166]:


a_mean,b_mean,c_mean=iris_class_mean_calc(a),iris_class_mean_calc(b),iris_class_mean_calc(c)
a_mean


# In[170]:


print(iris_class_varyans_and_standart_sapma_calc(a,a_mean))
print(iris_class_varyans_and_standart_sapma_calc(b,b_mean))
print(iris_class_varyans_and_standart_sapma_calc(c,c_mean))


# In[320]:


def get_mu_s():
    mu_0=[5,2,0]
    mu_1=[4,3.9,0]
    mu_2=[2,4,0]
    return mu_0,mu_1,mu_2


# In[321]:


def get_distance(mu,point):
    x=mu[0]-point[0]
    y=mu[1]-point[1]
    z=mu[2]-point[2]
    
    return ((x**2+y**2+z**2)**.5)


# In[322]:


my_flower_1=iris.data[0]
d_1=get_mu_s()[0]
get_distance(my_flower_1,d_1)


# In[341]:


def get_class_for_one_instance(flower):
    mu_s=get_mu_s()
    d_0=get_distance(mu_s[0],flower)
    d_1=get_distance(mu_s[1],flower)
    d_2=get_distance(mu_s[2],flower)
    
    if(d_0<d_1 and d_0<d_2):
        return 0
    elif(d_1<d_0 and d_1<d_2):
        return 1
    elif(d_2<d_1 and d_2<d_0):
        return 2


# In[349]:


def my_f_1(s):
    x=iris.data[s][0]
    y=iris.data[s][1]
    z=iris.data[s][2]
    my_f_1=(x,y,z) 
    r=get_class_for_one_instance(my_f_1)
    return r
for i in range(150):
    my_f1(i)


# In[384]:


def get_flower(s):
    x=iris.data[s][0]
    y=iris.data[s][1]
    z=iris.data[s][2]
    return [x,y,z] 
    
def update_mu():
    hata="none"
    mu_0_counter=1
    mu_0_sum=[0,0,0]
    mu_1_counter=1
    mu_1_sum=[0,0,0]
    mu_2_counter=1
    mu_2_sum=[0,0,0]
    
    for i in range(150):
        my_flower_data=get_flower(i)
        f_class=get_class_for_one_instance(my_flower_data)
        hata="exist"
        if(f_class==0):
            mu_0_counter=mu_0_counter+1
            mu_0_sum=[mu_0_sum[0]+my_flower_data[0],mu_0_sum[1]+my_flower_data[1],mu_0_sum[2]+my_flower_data[2]]
        if(f_class==1):
            mu_1_counter=mu_1_counter+1
            mu_1_sum=[mu_1_sum[0]+my_flower_data[0],mu_1_sum[1]+my_flower_data[1],mu_1_sum[2]+my_flower_data[2]]
        if(f_class==2):
            mu_2_counter=mu_2_counter+1
            mu_2_sum=[mu_2_sum[0]+my_flower_data[0],mu_2_sum[1]+my_flower_data[1],mu_2_sum[2]+my_flower_data[2]]
    mu_s=[]
    mu_s.append([mu_0_sum[0]/mu_0_counter,mu_0_sum[1]/mu_0_counter,mu_0_sum[2]/mu_0_counter])
    mu_s.append([mu_1_sum[0]/mu_1_counter,mu_1_sum[1]/mu_1_counter,mu_1_sum[2]/mu_1_counter])
    mu_s.append([mu_2_sum[0]/mu_2_counter,mu_2_sum[1]/mu_2_counter,mu_2_sum[2]/mu_2_counter])
    print(mu_s)
    return mu_s


# In[385]:


epoch=20
for i in range(epoch):
    mu_s=update_mu()
#while hata"none":
        #mu_s,hata=update_mu()


# In[ ]:




