"""Unit tests for the tracking module.

Implement your tracking module so that all tests pass.
An example of how to run these tests is given in run_tests.sh.

The tests below will be used to test the correctness of your
implementation.

You should add additional detailed tests as you add more
of your own functions to your implementation!
"""

import cv2
import extra_credit
import numpy
import unittest


class TestTracking(unittest.TestCase):
    """Tests the functionality of the tracking module."""

    def setUp(self):
        """Initializes shared state for unit tests."""
        pass

    def track_people_with_function_(self, video_filename, tracking_function):
        """Tracks the ball in 'video_filename' with 'tracking_function'."""
        video = cv2.VideoCapture(video_filename)
        tracking_function(video)

    def test_multi_tracking(self):
        """Tests for multi tracking."""
        self.track_people_with_function_(
            "seq_hotel.avi", extra_credit.multi_tracking)

if __name__ == '__main__':
    unittest.main()
