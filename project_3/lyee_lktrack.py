import cv2
import numpy as np

lk_params = dict(winSize = (15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10 ,.03))
subpix_params = dict(zeroZone=(-1,-1), winSize = (10,10), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20 ,.03))
feature_params = dict(maxCorners = 500, qualityLevel=0.01, minDistance =10)

class LKTracker(object):
	""" Class for Lucas-Kanade tracking with pyramidal optical flow. """

	def __init__(self, imnames):
		""" Initialize with a list of image names. """
		self.imnames = imnames
		self.features = []
		self.tracks = []
		self.current_frame = 0

	def detect_points(self):
		""" Detect 'good features to track' in the current frame """

		#load the image and converto to grayscale
		self.image = cv2.imread(self.imnames[self.current_frame])
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

		features = cv2.goodFeaturesToTrack(self.gray, **feature_params)

		cv2.cornerSubPix(self.gray, features, **subpix_params)

		#self.features = features
		self.features = self.use_contour_features
		self.tracks = [[p] for p in features.reshape((-1,2))]

		self.prev_gray = self.gray
		
	def use_contour_features(self):
		""" Use the contours from the square background as
		features instead of the Harris corner stuff """
		features = []
		
		im = cv2.imread(self.imnames[self.current_frame])

		im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(im2,127,255,0)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		for i in range(0, len(contours)):
			cnt = contours[i]
			x,y,w,h = cv2.boundingRect(cnt)
			features.append([[x,y]])		
		
		features = np.array(features, dtype='float32')
		print features
		print "============================"
		features = features.reshape(-1,1,2)
		print features
		self.features = features
		return features

	def track_points(self) :
		""" Track the detected features """
		if self.features != []:
			self.step()

			self.image = cv2.imread(self.imnames[self.current_frame])
			self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

			tmp = np.float32(self.features).reshape(-1,1,2)
			
			self.use_contour_features() # using the contours instead!
			#tmp = 

			features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, tmp, None, **lk_params)

			self.features = [p for (st,p) in zip(status, features) if st]

			features = np.array(features).reshape((-1,2))
			for i, f in enumerate(features):
				self.tracks[i].append(f)

			ndx = [i for (i, st) in enumerate(status) if not st]
			ndx.reverse()
			for i in ndx:
				self.tracks.pop(i)

			self.prev_gray = self.gray


	def step(self, framenbr = None):
		""" Step to another frame. If no argument is given, step to the next frame """
		if framenbr is None:
			self.current_frame = (self.current_frame +1) % len(self.imnames)
		else:
			self.current_frame = framenbr % len(self.imnames)

	def draw(self) :
		""" Draw the current image with points using OpenCV's own drawing functions."""

		# draw points as green circles
		for point in self.features:
			cv2.circle(self.image, (int(point[0][0]), int(point[0][1])),3,(0,255,0), -1)

		cv2.imshow('LKtrack', self.image)
		cv2.waitKey()
