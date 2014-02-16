__author__ = 'Eric'
import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('messi5.jpg',0)
#edges = cv2.Canny(img,100,200)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()



cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
edges = frame
while rval:
    cv2.imshow("preview", edges)
    rval, frame = vc.read()
    edges = cv2.Canny(frame,100,200)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")