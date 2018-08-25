from ppt.pointPerspectiveTransform import fourPointTransform
from skimage.filters import threshold_local
from PIL import Image
import numpy as np
import argparse
import cv2
import imutils
import os
import sys
import pytesseract

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to image to be scanned")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
ratio=image.shape[0]/500.0
orig=image.copy()
image=imutils.resize(image,height=500)


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200)

print("Image Detection")
cv2.imshow("Image",image)
cv2.imshow("Edged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


cnts=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[0] if imutils.is_cv2() else cnts[1]
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]


for c in cnts:
	peri =cv2.arcLength(c,True)
	approx=cv2.approxPolyDP(c,0.02*peri,True)

	#if len(approx)==4:
	screenCnt = approx
	break

print("Find Contours of paper")
cv2.drawContours(image, [screenCnt],-1,(0,255,0),2)
cv2.imshow("Outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()	


warped=fourPointTransform(orig,screenCnt.reshape(4,2)*ratio)
warped=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T=threshold_local(warped,11,offset=10,method="gaussian")
warped=(warped>T).astype("uint8")*255

print("Apply Perspective Transform")
cv2.imshow("Original",imutils.resize(orig,height=650))
cv2.imshow("Scanned",imutils.resize(warped,height=650))
cv2.waitKey(0)

#logo=Image.fromarray(warped)
#logo.save("warped.png")

#os.popen('tesseract warped.png stdout')
config=('--tessdata-dir "tessdata" -l Devanagari --oem 1 --psm 3' )

text= pytesseract.image_to_string(warped,config=config)

print(text)

f=open("scan.txt","w")
f.write(text)
f.close()

cv2.waitKey(0)
cv2.destroyAllWindows()










