__author__ = 'Eric'
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

MIN_MATCH_COUNT = 10

img1 = cv2.imread('..\\sample-images\\mainmenu.PNG',0)          # queryImage
img2 = cv2.imread('..\\sample-images\\flappybird-logo.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
avg_x = None # probably not the best
avg_y = None # ditto
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    print np.std(src_pts)
    std_dvt = np.std(src_pts)
    print np.mean(src_pts)
    mean_xy =  src_pts.mean(0)
    print mean_xy
    mx_pts = mean_xy[0,0]
    my_pts = mean_xy[0,1]
    mx_pts = int(mx_pts)
    my_pts = int(my_pts)
    top_left = (mx_pts + 20, my_pts + 20)
    bottom_right = (top_left[0] - 30, top_left[1] - 30)
    cv2.rectangle(img1,top_left, bottom_right, 0, -1)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    print len(src_pts)
    print len(matchesMask)

    #matches mask has flag for each src_pts if 1 draw if 0 not draw
    #use match mask

    #need better method for initial var value
    min_x = 1000000000000
    min_y = 1000000000000
    max_x = None
    max_y = None
    matchmask_idx = 0 # bad var name
    for xy_pts in src_pts:
        if matchesMask[matchmask_idx] == 1:
            #print "draw point"

            #print xy_pts
            x_pts = xy_pts[0,0]
            y_pts = xy_pts[0,1]
            x_pts = int(x_pts)
            y_pts = int(y_pts)
            max_x, discard_x = [max(x, 0) for x in [x_pts,max_x]]
            min_x, discard_x = [min(x, 1000000000) for x in [x_pts,min_x]]
            max_y, discard_y = [max(x, 0) for x in [y_pts,max_y]]
            min_y, discard_y = [min(x, 1000000000) for x in [y_pts,min_y]]
            #

            # we got values convert into bounding box
            top_left = (x_pts + 20, y_pts + 20)
            bottom_right = (top_left[0] - 30, top_left[1] - 30)
            #cv2.rectangle(img1,top_left, bottom_right, 255, -1)
        matchmask_idx += 1 # next value


        #dist = math.hypot(mx_pts - x_pts, my_pts - y_pts)
        #if dist < std_dvt:
        #    x_pts = int(x_pts)
        #    y_pts = int(y_pts)
    top_left = (min_x, min_y)
    bottom_right = (max_x, max_y)
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
