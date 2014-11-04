"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
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
    frames = []
    while 1:
        _, frame = video.read()

        if frame is None:
            break
        else:
            frames.append(frame)
    video.release()
    return frames


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
    frames = readFrames(video)

    # get background with background estimator method
    background = findBackground(frames)
    # Setup background subtractor object with parameters
    fgbg = cv2.BackgroundSubtractorMOG(30, 10, 0.7, 0)
    # Feed estimated background as first input to subtractor
    fgbg_init = fgbg.apply(background)

    # Iterate over every frame in video
    i = 0
    while i < len(frames):
        frame = frames[i]  # get the new frame

        frame = fgbg.apply(frame)  # apply background subtraction to frame

        # find contours in the frame
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # sort the contours in reverse order by width to eliminate small
        # contours caused by noise.
        sort_cnt = sorted(contours, key=lambda x: cv2.boundingRect(x)[2],
                          reverse=True)

        # get the parameters of the bounding box for the largest width contour
        x, y, w, h = cv2.boundingRect(sort_cnt[0])
        # append to result list
        result.append((x, y, x + w, y + h))

        # Comment these lines back in to see result with bounding box
        # orig_frame = frames[i]
        # cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('frame', orig_frame)
        # cv2.waitKey(30)

        i += 1

    cv2.destroyAllWindows()
    return result


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    return track_ball_1(video)


def track_ball_3(video):
    """As track_ball_1, but for ball_3.mov."""
    return track_ball_1(video)


def track_ball_4(video):
    """As track_ball_1, but for ball_4.mov."""

    result = []
    frames = readFrames(video)

    i = 0
    while i < len(frames):
        frame = frames[i]  # get the new frame

        # find edges of the ball in the frame
        # parameters filter out background
        edges = cv2.Canny(frame, 700, 800)

        # find contours in the edge frame
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # sort the contours in reverse order by width to eliminate small
        # contours caused by noise.
        sort_cnt = sorted(contours, key=lambda x: cv2.boundingRect(x)[2],
                          reverse=True)

        # get the parameters of the bounding box for the largest width contour
        x, y, w, h = cv2.boundingRect(sort_cnt[0])
        # append to result list
        result.append((x, y, x + w, y + h))

        # Comment these lines back in to see result with bounding box
        # orig_frame = frame[i]
        # cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("ball", orig_frame)
        # cv2.waitKey(30)

        i += 1

    cv2.destroyAllWindows()
    return result


def track_face(video):
    """As track_ball_1, but for face.mov."""

    # Get the cascade classifier with pre-trained classifiers
    face_cascade = cv2.CascadeClassifier('frontal_face.xml')

    # initialize a previous face with zeros
    prev_face = (0, 0, 0, 0)
    result = []

    # Loop until frame from video capture returns None
    while(1):
        _, img = video.read()
        if img is None:
            break

        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Do the face detection. Set minNeighbors relatively high
        # to eliminate false positive faces.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5,
                                              cv2.cv.CV_HAAR_SCALE_IMAGE)

        # If the number of faces found is 0, use face from previous frame
        # as a good indicator of where face in the current frame should be.
        # Has inherent error but works well.
        if len(faces) == 0:
            faces = [prev_face]

        # Sort faces by width of bounding box in reverse order
        sort_face = sorted(faces, key=lambda x: x[2],
                          reverse=True)

        # Get bounding box of largest face, assumes smaller ones are
        # false positives.
        x, y, w, h = sort_face[0]

        # append to results
        result.append((x, y, x + w, y + h))
        # set previous face
        prev_face = faces[0]

        # Uncomment these lines to see results
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y: y + h, x: x + w]
        # roi_color = img[y: y + h, x: x + w]
        # cv2.imshow('img', img)
        # cv2.waitKey(30)

    return result
