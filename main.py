from PIL import Image
from crnn.crnn_torch import crnnOcr
#from crnn.crnn_torch import crnnOcr_number
import numpy as np
import time
import os
import cv2
import detect
import time
def address(img):
    h,w=img.shape[:2]
    if h/2 >5:
        cut = 2
    else:
        cut = 3
    if cut==2:
        h1 = int((h-1 )/2)
        copy = np.zeros((h1,2*w,3))
        copy = np.uint8(copy)
        copy[0:h1,0:w] = img[0:h1,0:w]
        copy[0:h1,w:2*w] = img[h1:2*h1,0:w]
    else:
        h1 = int(h /3)
        copy = np.zeros(h1,3*w,3)
        copy = np.uint8(copy)
        copy[0:h1,0:w] = img[0:h1,0:w]
        copy[0:h1,w:2*w] = img[h1:2*h1,0:w]
        copy[2*w:3*w,0:h1,] = img[2*h1:3*h1,0:w]
    cv2.imwrite('2.jpg',copy)
    return copy
def Crnn():

    result = {'1':'','2':'','3':'','4':'','5':'','6':'','7':'','7':''}
    region = detect.return_result()
    print('\n',"*************结果显示*****************")
    for f in region.keys():
        if f ==4:
            region[f] = address(region[f])
        img = Image.fromarray(cv2.cvtColor(region[f],cv2.COLOR_BGR2RGB)).convert('L')
        res = crnnOcr(img)
        
        print(res)
    return result
t0 = time.time()
a = Crnn()
print(time.time()-t0) 
