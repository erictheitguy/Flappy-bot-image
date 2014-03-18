__author__ = 'Eric'

#open web cam and take pictures


import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
# set camera resolution
vc.set(3,1280)
vc.set(4,720)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
i = 0

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    if key == 0x63 or key == 0x43: # c key
        print 'capturing!'
        # should create the file name as a var
        # puts image in same directory as file
        cv2.imwrite("capture"+str(i)+".jpg",frame)


        i+=1

cv2.destroyWindow("preview")
