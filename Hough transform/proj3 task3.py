import numpy as np
import imageio
import math
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('D:\\hough.jpg',0)
width, height = img.shape
diaglen = int(round(math.sqrt(width * width + height * height)))

def hough_line(img):
    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1))
    width, height = img.shape
    diaglen = int(round(math.sqrt(width * width + height * height)))
    rs = np.linspace(-diaglen, diaglen, diaglen * 2)
    cost = np.cos(thetas)
    sint = np.sin(thetas) 
    accumulator = np.zeros((2 * diaglen, len(thetas)), dtype=np.uint8)
    y_id, x_id = np.nonzero(img)
    for i in range(len(x_id)):
        x = x_id[i]
        y = y_id[i]
        for teta in range(len(thetas)):
            r = diaglen + int(round(x * cost[teta] + y * sint[teta]))
            accumulator[r, teta] += 1
    return accumulator, thetas, rs

def sobel(img):
# define gaussian kernels
    ksv=[[1,0,-1],[2,0,-2],[1,0,-1]]
    ksh=[[1,2,1],[0,0,0],[-1,-2,-1]]

###################sobel edge detection######################################
# function for convolution.
    def convolu(ker):
        ksf=[[0 for j in range(len(ker[0]))] for i in range(len(ker))]  
    
#flipping the kernel
        for i in range(len(ker)):
            for j in range(len(ker[0])):
                ksf[i][j]=ker[len(ker)-i-1][len(ker)-j-1] 
            
#convolution            
        res=[[0 for j in range(len(gray[0]))] for i in range(len(gray))]
        kh=len(ksf)//2
        kw=len(ksf[0])//2
        ih=len(gray)
        iw=len(gray[0])
        for i in range(kh,ih-kh):
            for j in range(kw,iw-kw):
                x=0
                for l in range(len(ksf)):
                    for k in range(len(ksf[0])):
                        x = x+ gray[i-kh+l][j-kw+k]*ksf[l][k]      
                res[i][j]=x     
        return res   

#method 2 for eliminating zeros
    def eliminate2(resl):
        for i in range(len(resl)):
            resl[i]=[abs(j) for j in resl[i]]
        maximum = max([max(j) for j in resl])
        for i in range(len(resl)):
            resl[i][:] = [x / maximum for x in resl[i]]
        return resl
    gray = img 
    reslh=convolu(ksh) 
    reslv=convolu(ksv)
    reslho = eliminate2(reslh)
    reslho = np.asarray(reslho)
    reslve = eliminate2(reslv)  
    reslve = np.asarray(reslve)
    for i in range(len(reslve)):    
        for j in range(len(reslve[0])):
            if(reslve[i][j]>0.1):
                reslve[i][j] = 1
            else:
                reslve[i][j] = 0
    return reslve,reslho

###################MAIN FOR LINES#################################################################
accu = []
imgpath = 'D:\\hough.jpg'
tran = imageio.imread(imgpath) 
tran1 = imageio.imread(imgpath)
img = cv2.imread('D:\\hough.jpg',0) #read image as as gray
img,reslho = sobel(img)
cv2.imshow('imcan.png',img)      
accumulator, thetas, rhos = hough_line(img)
print(accumulator)
print('over')
print(img.shape)
for i in range(len(accumulator)):
    for j in range(len(accumulator[0])):
        if(accumulator[i][j]>195):
            accu.append([thetas[j],rhos[i]])
accu = np.asarray(accu)
print(accu)

for theta,r in accu: 
    a = np.cos(theta) 
    b = np.sin(theta) 
    x0 = a*r 
    y0 = b*r 
    x1 = int(x0 + 2000*(-b))  
    y1 = int(y0 + 2000*(a)) 
    x2 = int(x0 - 2000*(-b))   
    y2 = int(y0 - 2000*(a))   
#    thetamap = np.deg2rad(theta)
    if(-2.10865238e-01<=theta<=2.10865238e-01):
        cv2.line(tran,(x1,y1), (x2,y2), (0,0,255),2)
    else:
        cv2.line(tran1,(x1,y1), (x2,y2), (255,0,0),2)
            
cv2.imwrite('D:\\red_lines.jpg', tran) 
cv2.imwrite('D:\\blue_lines.jpg', tran1)


############################# CIRCLE DETECTION ######################################################################

def detectCircles(img,threshold,region,radius = None):
    (M,N) = img.shape
    if radius == None:
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius

    R = R_max - R_min
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                               
    for val in range(R):
        r = R_min+val
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                       
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       
            X = [x-m+R_max,x+m+R_max]                                           
            Y= [y-n+R_max,y+n+R_max]                                            
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0

    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1

    return B[:,R_max:-R_max,R_max:-R_max]

def displayCircles(A):
    img = cv2.imread(file_path)
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)             
    circle = []
    for r,x,y in circleCoordinates:
        if(r<25 and r>17):
            circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
            fig.add_subplot(111).add_artist(circle[-1])
    plt.show()
    plt.savefig('D:\\coin.jpg')
    
############################################main for circles#########################################    
file_path = 'D:\\hough.jpg'
res = img    
for i in range(len(reslho)):    
    for j in range(len(reslho[0])):
        if(reslho[i][j]>0.1):
            res[i][j] = 1
        else:
            res[i][j] = 0

res = detectCircles(res,11,20,radius=[50,15])
displayCircles(res)

