"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy as np


def findBackground(video):

    _, background = video.read()

    i = 1
    n = 100  # Number of frames to use for background estimation

    # Read each frame and do a weighted sum
    while(i < n):
        i += 1
        _, frame = video.read()
        beta = 1.0/i
        alpha = 1.0 - beta
        background = cv2.addWeighted(background, alpha, frame, beta, 0.0)

    # reset video capture to first frame
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)

    cv2.imshow('estimated background', background)
    cv2.waitKey(3000)

    return background


def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
      video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
      a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
      coordinates of the rectangular bounding box of the ball in each frame.
    """

    result = []

    background = findBackground(video)
    fgbg = cv2.BackgroundSubtractorMOG(30, 10, 0.7, 0)
    fgbg_init = fgbg.apply(background)

    while(1):
        _, frame = video.read()
        if frame is None:
            break

        orig_frame = frame
        frame = fgbg.apply(frame)

        contours, _ = cv2.findContours(frame, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        sort_cnt = sorted(contours, key=lambda x: cv2.boundingRect(x)[2],
                          reverse=True)

        x, y, w, h = cv2.boundingRect(sort_cnt[0])
        result.append((x, y, x+w, y+h))
        cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('frame', orig_frame)
        cv2.waitKey(30)

    cv2.destroyAllWindows()
    return result


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    return track_ball_1(video)


def track_ball_3(video):
    """As track_ball_1, but for ball_2.mov."""
    return track_ball_1(video)


def track_ball_4(cap):
    

# take first frame of the video
    _,frame = cap.read()

# setup initial location of window
    r,h,c,w = 250,90,400,125  # simply hardcoded the values
    track_window = (c,r,w,h)

# set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while(1):
        ret ,frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image

            #pts = cv2.boxPoints(ret)
            #pts = np.int0(pts)
            #img2 = cv2.polylines(frame,[pts],True, 255,2)
            contours, _ = cv2.findContours(frame, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

            sort_cnt = sorted(contours, key=lambda x: cv2.boundingRect(x)[2],
                          reverse=True)

            x, y, w, h = cv2.boundingRect(sort_cnt[0])
            result.append((x, y, x+w, y+h))
            img2 = cv2.rectangle(track_window, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('img2',img2)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k)+".jpg",img2)

        else:
            break

    cv2.destroyAllWindows()
    cap.release()

def track_face(video):
    """As track_ball_1, but for face.mov."""
    pass
