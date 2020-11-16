import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
import skimage as ski

def images():
    z=[]
    for i in os.listdir("./images"):
        z.append(i)
    return z

def displayImage(image, title):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

#input image and color to be excluded
def rgbExclusion(image, excludeColor):
    if excludeColor=="B":
        imageRG=image.copy()
        imageRG[:,:,0]=0
        return imageRG
    elif excludeColor=="G":
        imageRB=image.copy()
        imageRB[:,:,1]=0
        return imageRB
    elif excludeColor=="R":
        imageGB=image.copy()
        imageGB[:,:,2]=0
        return imageGB
        
def displayHistogram(image,title):
    plt.hist(image.ravel(),256,[0,256])
    plt.title(title)
    plt.show()


def Convolution2d(image, kernel):
    zeroPadding=1 #Zero Padding in the image
    kernel = np.flipud(np.fliplr(kernel)) # Flipping Kernel
    convolvedImage = np.zeros_like(image)            # convolved Image
    
    # Add zero padding to the input image
    paddedImage = np.zeros((image.shape[0] + zeroPadding*2, image.shape[1] + zeroPadding*2))   
    paddedImage[int(zeroPadding):int(-1 * zeroPadding), int(zeroPadding):int(-1 * zeroPadding)] = image
    
    # element wise multiplication and summation
    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(image.shape[1]):
            convolvedImage[x,y]=(kernel*paddedImage[x:x+kernel.shape[0],y:y+kernel.shape[1]]).sum()
    return convolvedImage
    
def displayTwoImages(image1, title1, image2, title2):
    f,a=plt.subplots(1, 2)
    a[0].imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    a[0].set_title(title1)
    a[1].imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    a[1].set_title(title2)
    plt.show()
    
def displayThreeImages(image1, title1, image2, title2, image3, title3):
    f,a=plt.subplots(1, 3)
    a[0].imshow(image1)
    a[0].set_title(title1)
    a[1].imshow(image2)
    a[1].set_title(title2)
    a[2].imshow(image3)
    a[2].set_title(title3)
    plt.show()

