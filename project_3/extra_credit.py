"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""
"""Project 3: Tracking.

Extra credit - multi tracking pedestrians
"""

import cv2
import numpy as np


def findBackground(frames):
    """ Reads n number of frames and uses a weighted average
        to estimate the background of the video.

        Arguments:
          frames: a list of consecutive image frames from a video

        Outputs:
          A single image representing the estimated background of the video
    """

    n = 100  # Number of frames to use
    i = 1
    background = frames[0]
    # Read each frame and do a weighted sum
    # beta is the weight for each new frame which starts at 1/2 and
    # gets smaller
    # alpha is the weight of the sum and starts at 1/2 and gets larger
    while(i < n):
        i += 1
        frame = frames[i]
        beta = 1.0 / (i + 1)
        alpha = 1.0 - beta
        background = cv2.addWeighted(background, alpha, frame, beta, 0.0)

    # Comment these lines back in to see result
    # cv2.imshow('estimated background', background)
    # cv2.waitKey(3000)

    return background


def readFrames(video):
    """ Simply reads all the frames from video and
        stores them in a list which is returned.
        We do this so that we can read some number
        of frames to use in background estimation
        without having to reset the video capture.

        Arguments:
          video: an open cv2.VideoCapture object

        Outputs:
          A list of all the frames in consecutive order from the video
    """
    cv2.namedWindow("input")
    frames = []
    while 1:
        _, frame = video.read()
        if frame is None:
            break
        else:
            frames.append(frame)
    video.release()
    return frames


def multi_tracking(video):
    print "starting"
    
    result = []
    frames = readFrames(video)
    print "done reading frames"

    # get background with background estimator method
    background = findBackground(frames)
    print "done finding background"
    
    # Setup background subtractor object with parameters
    fgbg = cv2.BackgroundSubtractorMOG(30, 10, 0.7, 0)
    # Feed estimated background as first input to subtractor
    fgbg_init = fgbg.apply(background)
    # Iterate over every frame in video
    i = 0
    while i < len(frames):
        frame = frames[i]  # get the new frame

        frame = fgbg.apply(frame)  # apply background subtraction to frame

        cv2.imshow('frame', frame)
        cv2.waitKey(30)

        i += 1

    cv2.destroyAllWindows()
    return result

video = cv2.VideoCapture("seq_hotel.avi")
multi_tracking(video)
