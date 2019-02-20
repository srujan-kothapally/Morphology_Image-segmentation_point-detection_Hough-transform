import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib.pyplot as plt
threshold_values = {}
h = [1]

def Hist(img):
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   return y

def regenerate_img(img, threshold):
    row, col = img.shape 
    u = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                u[i,j] = 255
            else:
                u[i,j] = 0
    return u
Equiv={}
for i in range(1000):
    Equiv[i]=i
def connectedcomponents(image):
    i=cv2.imread('H:/cvip projects/proj3/original_imgs/original_imgs/segment.jpg')
    i1=np.asarray(i)
    print(i1)
    count=0
    Temp=np.zeros(image.shape)
    Equivalent=np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j]==0):
                Temp[i][j]=0
            else:
                if(Temp[i-1][j]==0 and Temp[i][j-1]==0):
                    count=count+1
                    Temp[i][j]=count
                if(Temp[i-1][j]!=0 and Temp[i][j-1]==0):
                    Temp[i][j]=Temp[i-1][j]
                if(Temp[i-1][j]==0 and Temp[i][j-1]!=0):
                    Temp[i][j]=Temp[i][j-1]
                if(Temp[i-1][j]!=0 and Temp[i][j-1]!=0):
                    if(Temp[i-1][j]!=Temp[i][j-1]):
                        Temp[i][j]=min(Temp[i][j-1],Temp[i-1][j])
                        z=max(Temp[i][j-1],Temp[i-1][j])
                        Equiv[z]=min(Temp[i][j-1],Temp[i-1][j])
                    else:
                        Temp[i][j]=min(Temp[i][j-1],Temp[i-1][j])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(Temp[i][j]!=0):
                label=Temp[i][j]
                Equivalent[i][j]=int(Equiv[label])
    labels=[0]
    Total_components=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flag=0
            for k in range(len(labels)):
                if(Equivalent[i][j]==labels[k]):
                    flag=1
            
            if(flag==0):
                labels.append(Equivalent[i][j])
    Total_components=len(labels)-1
    labels_count=[0 for i in range(len(labels))]
    
#print(Equivalent[1:200])
    Eq2=np.copy(np.asarray(Equivalent))
    fi_labels={}
    for i in range(225):
        fi_labels[i]=i
    #print(fi_labels)
#    Eq2=Equivalent
    print(np.max(Eq2))
    print('hi')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            fi_labels[Eq2[i][j]]=fi_labels[Eq2[i][j]]+1
    fi_labels[0]=0
    for i in range(225):
        if(fi_labels[i]<200):
            fi_labels[i]=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(fi_labels[Eq2[i][j]]==0):
                Eq2[i][j]=0
            #Eq2[i][j]=int(fi_labels[Eq2[i][j]])
            #print(Eq2[i][j])
            #print(fi_labels[Eq2[i][j]])
    T2=0
    l2=[0]
    print('hello',np.max(Eq2))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flag=0
            for k in range(len(l2)):
                if(Eq2[i][j]==l2[k]):
                    flag=1
            
            if(flag==0):
                l2.append(Eq2[i][j])
    T2=len(l2)-1
    coordinates=[[[1000,1000],[0,0]] for i in range(221)]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(Eq2[i][j]!=0):
                Eq2[i][j]=(Eq2[i][j])//1
                #print(Eq2[i][j])
                a=np.int(Eq2[i][j])
                if(coordinates[a][0][0]>i):
                    
                    coordinates[a][0]=[i,j]
                if(coordinates[a][1][1]<j):
                    coordinates[a][1]=[i,j]
    
    for h in range(len(coordinates)):
        if(coordinates[h]!=[[1000,1000],[0,0]]):
            print(coordinates[h])
            
            x1=coordinates[h][0][0]
            y1=coordinates[h][0][1]
            x2=coordinates[h][1][0]
            y2=coordinates[h][1][1]
            #print(x1,y1,x2,y2)
            #print(i1[x1][y1])
            i1[x1,y1]=[0,255,0]
            i1[x2,y2]=[255,0,0]
            #print(i1[x1][y1])
            cv2.rectangle(i1,(y1,x1),(y2,x2), (0, 255, 0), 3)
    cv2.imshow('ff',np.asarray(i1))
    cv2.waitKey(0)
    print(T2)     
    print(Total_components)
    #print(labels)
    #print(Equivalent)'''
    return(Equivalent)


img = cv2.imread('D:\\segment.jpg',0)
img=cv2.GaussianBlur(img,(5,5),0)
im = cv2.imread('D:\\segment.jpg')
h = Hist(img)
res = regenerate_img(img, 195)
cv2.rectangle(im,(381,32),(450,260), (0,255,255), thickness=2, lineType=8, shift=0)
c = cv2.rectangle(im,(163,121),(210,164), (0,255,255), thickness=2, lineType=8, shift=0)
c = cv2.rectangle(im,(247,73),(307,204), (0,255,255), thickness=2, lineType=8, shift=0)
c = cv2.rectangle(im,(329,20),(370,290), (0,255,255), thickness=2, lineType=8, shift=0)
cv2.imwrite('D:\\box',c)

