"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy


def findBackground(video):

    _, background = video.read()

    i = 1
    n = 100  # Number of frames to use for background estimation

    # Read frames and do a weighted sum
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


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""
    pass

def track_face(video):
    """As track_ball_1, but for face.mov."""
    pass
