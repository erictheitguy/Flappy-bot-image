__author__ = 'Eric'
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

MIN_MATCH_COUNT = 10

img1 = cv2.imread('capture2.jpg',0)          # queryImage
img2 = cv2.imread('bird2.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 5
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 50)
search_params = dict(checks = 500)

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

    #matches mask has flag for each src_pts if 1 draw if 0 not draw
    #use match mask

    #need better method for initial var value
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
    print top_left,bottom_right
    cv2.rectangle(img1,top_left, bottom_right, 255, -1)
        #print x_pts
        #print y_pts





else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)



img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
