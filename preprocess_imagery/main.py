import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def histograEqualColor(img_bgr):
    img_ycrcb=cv.cvtColor(img_bgr,cv.COLOR_BGR2YCrCb)
    y=img_ycrcb[:,:,0]

    hist=cv.calcHist([y],[0],None,[256],[0,256])
    cdf=hist.cumsum()
    cdfNorm=cdf*float(hist.max())/cdf.max()

    plt.figure()
    plt.subplot(231)
    plt.imshow(cv.cvtColor(img_bgr,cv.COLOR_BGR2RGB))
    plt.title("Oryginal")

    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm,color='b')

    y_eq=cv.equalizeHist(y)
    img_ycrcb_eq=img_ycrcb.copy()
    img_ycrcb_eq[:,:,0]=y_eq
    img_eq_bgr=cv.cvtColor(img_ycrcb_eq,cv.COLOR_YCrCb2BGR)

    equhist=cv.calcHist([y_eq],[0],None,[256],[0,256])
    eqycdf=equhist.cumsum()
    eqycdfNorm=eqycdf*float(equhist.max())/eqycdf.max()

    plt.subplot(232)
    plt.imshow(cv.cvtColor(img_eq_bgr,cv.COLOR_BGR2RGB))
    plt.title("Equalizacja")

    plt.subplot(235)
    plt.plot(equhist)
    plt.plot(eqycdfNorm,color='b')

    clahe=cv.createCLAHE(clipLimit=5,tileGridSize=(8,8))
    y_clahe=clahe.apply(y)

    img_ycrcb_clahe=img_ycrcb.copy()
    img_ycrcb_clahe[:,:,0]=y_clahe
    img_clahe_bgr=cv.cvtColor(img_ycrcb_clahe,cv.COLOR_YCrCb2BGR)

    clahehist=cv.calcHist([y_clahe],[0],None,[256],[0,256])
    clahecdf=clahehist.cumsum()
    clahecdfNorm=clahecdf*float(clahehist.max())/clahecdf.max()

    plt.subplot(233)
    plt.imshow(cv.cvtColor(img_clahe_bgr,cv.COLOR_BGR2RGB))
    plt.title("CLAHE")

    plt.subplot(236)
    plt.plot(clahehist)
    plt.plot(clahecdfNorm,color='b')    


    plt.show()
    pass

if __name__ == "__main__":
    root = os.getcwd()
    imgpath = os.path.join(root,"datasets/UAV_VisLoc_dataset/02/drone/02_0071.JPG")
    img = cv.imread(imgpath)
    histograEqualColor(img)
