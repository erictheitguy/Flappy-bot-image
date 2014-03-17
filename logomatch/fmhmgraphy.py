__author__ = 'Eric'
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(3,1280)
vc.set(4,720)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

train_image = cv2.imread('..\\sample-images\\neg26.jpg',0) #loaded image as gray
img1 = frame # video image
gray_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) #convert video to gray
# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray_img,None)
kp2, des2 = sift.detectAndCompute(train_image,None) # we really only need this once

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
mask = None
M = None

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img1 = cv2.polylines(img1,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,train_image,kp2,good,None,**draw_params)
good_M = M
good_mask = mask
while rval:
    cv2.imshow("preview", img3)
    rval, frame = vc.read()
    img1 = frame
    gray_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray_img,None)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #The follow is a workaround for a bug in the cv2.RANSAC method
        if mask is None:
            print "we going to crash"
            M, mask = cv2.findHomography(src_pts, dst_pts,0,5.0)
            # lets reuse a last good mask
            #mask = good_mask
            #M = good_M
            matchesMask = mask.ravel().tolist()
        else:
            matchesMask = mask.ravel().tolist()
            good_M = M
            good_mask = mask
        h,w = gray_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        #img1 = cv2.polylines(img1,[np.int32(dst)],True,255,15, cv2.LINE_AA)
        min_x = 1000000000
        min_y = 1000000000
        max_x = 0
        max_y = 0
        matchmask_idx = 0 # bad var name
        for xy_pts in src_pts:
            if matchesMask[matchmask_idx] == 1:
                #print "draw point"

                #print xy_pts
                x_pts = xy_pts[0,0]
                y_pts = xy_pts[0,1]
                x_pts = int(x_pts)
                y_pts = int(y_pts)
                if min_x > x_pts:
                    min_x = x_pts
                if min_y > y_pts:
                    min_y= y_pts
                if max_x < x_pts:
                    max_x = x_pts
                if max_y < y_pts:
                    max_y = y_pts
            matchmask_idx += 1 # next value



        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
        #print top_left,bottom_right
        cv2.rectangle(img1,top_left, bottom_right, 255, -1)
        print "Enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        #we would then call the next routine to start the game
        #now we have to look for the ready screen
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,train_image,kp2,good,None,**draw_params)
    #img3 = img1
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")



