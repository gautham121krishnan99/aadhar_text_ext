#!/usr/bin/env python
# coding: utf-8

# In[512]:


import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from numpy import array
from PIL import ImageEnhance, ImageFilter
import pytesseract
import string as s
import re
from typing import Tuple, Union
import math
from deskew import determine_skew
import argparse


# In[516]:



def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def crop(full_path,i):
    image = cv2.imread(full_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    im = rotate(image, angle, (0, 0, 0))
    cv2.imwrite("output.jpg", im)
    im=Image.open("output.jpg")
    im=im.convert('L')
    im=ImageEnhance.Contrast(im)
    im=im.enhance(5)
    im=im.filter(ImageFilter.EDGE_ENHANCE)
    im=im.resize((im.size[0],im.size[1]))
    im.save("output.jpg")
    img=cv2.imread("output.jpg")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel=np.ones((5,5),np.uint8)
    edges=cv2.Canny(gray,0,200,apertureSize=3)
    edges=cv2.dilate(edges,kernel)
    _,contours,heirarchy=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    area=[cv2.contourArea(c) for c in contours]
    Ar_max=np.argmax(area)
    cMax=contours[Ar_max]
    x,y,w,h=cv2.boundingRect(cMax)
    im=im.crop((x,y,x+w,y+h))
    im=im.rotate(90*i)
    im.save("cropped.jpeg")
    cropped=cv2.imread("cropped.jpeg")
    os.remove("cropped.jpeg")
    return cropped


def Contours(full_path,i):
    cropped=crop(full_path,i)
    gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    rec,thresh=cv2.threshold(gray,0,100,cv2.THRESH_BINARY_INV)
    dilation=cv2.dilate(thresh,kernel,iterations=1)
    dilation=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)
    _,contours,_=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours,gray,dilation

def OCR(file_name,i):
    pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    contours,cropped,dilation=Contours(file_name,i)
    ret,cropped=cv2.threshold(cropped,0,255,cv2.THRESH_OTSU)
    f=open("E:\\python\\output_files\\text.txt","w+")
    #cropped=cv2.GaussianBlur(cropped,(5,5),cv2.BORDER_DEFAULT)
    cv2.imwrite("temp.jpg",cropped)
    im=Image.open("temp.jpg")
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        piece=cropped[y:y+h,x:x+w]
        text=pytesseract.image_to_string(piece,config='-c tessedit_char_whitelist=0123456789')
        text=text.replace(" ","")
        f.write(text)
    f.close()
    cv2.drawContours(cropped,contours,-1,(10,255,5),2)
    return cropped,contours

def process(file_name):
    os.getcwd()
    path="E:/python/images"
    full_path=path+'/'+file_name
    i=3
    print("try orientation =",i)
    cropped,contours=OCR(full_path,i)
    f=open("E:\\python\\output_files\\text.txt")
    lines=f.readlines()
    f.close
    pattern="\d{12}"
    for line in lines:
        if re.match(pattern,line):
            text=line[0:12]
            print("aadhar number detected =",text)
    try:
        return cropped,contours,text
    except:
        try:
            i=2
            print("try orientation =",i)
            cropped,contours=OCR(full_path,i)
            f=open("E:\\python\\output_files\\text.txt")
            lines=f.readlines()
            f.close
            pattern="\d{12}"
            for line in lines:
                if re.match(pattern,line):
                    text=line[0:12]
                    print("aadhar number detected =",text)
            return cropped,contours,text
        except:
            try:
                i=1
                print("try orientation =",i)
                cropped,contours=OCR(full_path,i)
                f=open("E:\\python\\output_files\\text.txt")
                lines=f.readlines()
                f.close
                pattern="\d{12}"
                for line in lines:
                    if re.match(pattern,line):
                        text=line[0:12]
                        print("aadhar number detected =",text)
                return cropped,contours,text
            except:
                try:
                    i=0
                    print("try orientation =",i)
                    cropped,contours=OCR(full_path,i)
                    f=open("E:\\python\\output_files\\text.txt")
                    lines=f.readlines()
                    f.close
                    pattern="\d{12}"
                    for line in lines:
                        if re.match(pattern,line):
                            text=line[0:12]
                            print("aadhar number detected =",text)
                    return cropped,contours,text
                except:
                    print("aadhar number not detected")
                    text= None
                    return cropped,contours,text
    os.remove("temp.jpg")        


# In[518]:


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("inp_dir",help="image name with extension",type=str)
    args=parser.parse_args()
    default_input_path="E:\python\images"
    default_output_path="E:\python\Output_files"
    print("default input path =",default_input_path)
    print("\n")
    print("default_output_path=",default_output_path)
    print("\n")
    cropped,contours,text=process(args.inp_dir)
    cv2.imwrite(os.path.join(default_output_path,args.inp_dir),cropped)
    plt.imshow(cropped)
    
if __name__=='__main__':
    main()


# In[ ]:




