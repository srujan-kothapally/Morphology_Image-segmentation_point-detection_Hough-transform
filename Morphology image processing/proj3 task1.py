import numpy as np
import random
import cv2
img1 = cv2.imread("D:\\noise.JPG",0)
img2 = np.zeros((310,351))
imgd = np.zeros((310,351))
cv2.imshow('original',img1)
kernel = np.ones((3,3))
kernel[:][:]=255*kernel[:][:]
imginv = (255-img1)
def dilation(kernel,img1,img2):
    
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            if(kernel[i][j]==255):
                for m in range(310-1):
                    for n in range(351-1):
                        if(img1[m][n]==255):
                            img2[m+i-1][n+j-1]=255
    return img2
def erode(kernel,img1):
    img3 = np.zeros((311,352))
    imginv = (255-img1)
    ero = dilation(kernel,imginv,img3)
    erosion = (255-ero)
    return erosion
erosion1= erode(kernel,img1)
dilation1= dilation(kernel,img1,img2)

def closing(kernel,img1,img4,dilation1):
    erosion2 = erode(kernel,dilation1)
    return erosion2


def opening(kernel,img1,img2,erosion1):
    img10 = np.zeros((311,352))
    dilation4  = dilation(kernel,erosion1,img10)
    return dilation4
closing1 = closing(kernel,img1,img2,dilation1)
opening1 = opening(kernel,img1,img2,erosion1)  

def boundary(openin1,erosionop,erosioncl,closin1):
    print(openin1.shape)
    print(closin1.shape)
    print(erosionop.shape)
    print(erosioncl.shape)
    img11 = np.zeros((311,352))
    img12 = np.zeros((311,352))
    for i in range(len(img12)):
        img11[i] = openin1[i]-erosionop[i]
        img12[i] = closin1[i]-erosioncl[i]
 
    return img11,img12
erosionop= erode(kernel,opening1)
erosioncl= erode(kernel,closing1)
boundop,boundcl = boundary(opening1,erosionop,erosioncl,closing1)

cv2.imwrite('D:\\res_noise1.jpg',closing1)
cv2.imwrite('D:\\res_noise2.jpg',opening1)
cv2.imwrite('D:\\dilation.jpg',dilation1)
cv2.imwrite('D:\\erosion.jpg',erosion1) 
cv2.imwrite('D:\\res_bound1.jpg',boundcl)
cv2.imwrite('D:\\res_bound2.jpg',boundop) 
          


