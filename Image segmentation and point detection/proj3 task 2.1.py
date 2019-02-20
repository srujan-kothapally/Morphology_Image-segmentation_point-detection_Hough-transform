import numpy as np
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread("D:\\point.JPG",0)
img5 = cv2.imread("D:\\point.JPG",0)
img1 = cv2.GaussianBlur(img1,(5,5),0)
print(img1.shape)
img2 = np.zeros((475,357))
cv2.imshow('original',img1)
kernel = np.ones((3,3))
kernel[:][:] = -1
kernel[1][1]=8
print(kernel)
thres=0
for i in range(1,len(img1)-1):
    for j in range(1,len(img1[0])-1):
        for k in range(-1,2):
            for l in range(-1,2):
                z = img1[i+k][j+l]*kernel[k+1][l+1]
                thres+=z
        if(thres>=51 and thres<52):
            print(i)
            print(j)
            c = cv2.rectangle(img5,(271,139),(290,156), (0,255,255), thickness=1, lineType=8, shift=0)
            img2[i][j]=thres
            thres=0
        else:
            thres=0
cv2.imshow('detected_point',img2)
cv2.imwrite('D:\\original.jpg',c)   
        
                
