__author__ = 'Eric'
import cv2
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

#template all ready loaded as gray scale?
template = cv2.imread('sample-images\\ground-c1.png',0)
#gray_template = template.copy()
#gray_template = cv2.cvtColor(gray_template,cv2.COLOR_RGB2GRAY)
w, h = template.shape[::-1]



if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()

else:
    rval = False

img = frame
gray_img = img.copy()
gray_img = cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
while rval:
    cv2.imshow("preview", gray_img)

    rval, frame = vc.read()
    res = cv2.matchTemplate(gray_img,template,cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(gray_img,top_left, bottom_right, 255, -1)
    img = frame
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")