"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy


def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
      video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
      a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
      coordinates of the rectangular bounding box of the ball in each frame.
    """
    fgbg = cv2.createBackgroundSubtractorMOG()
    while(1):
      ret, frame = video.read()

      fgmask = fgbg.apply(frame)

      cv2.imshow('frame',fgmask)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break

    video.release()
    video.destroyAllWindows()


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    pass


def track_ball_3(video):
    """As track_ball_1, but for ball_2.mov."""
    pass


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""
    pass


def track_face(video):
    """As track_ball_1, but for face.mov."""
    pass
