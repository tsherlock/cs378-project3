import lktrack

imnames = ['frame_000.png', 'frame_001.png','frame_002.png','frame_003.png','frame_004.png','frame_005.png','frame_006.png','frame_007.png']


lkt = lktrack.LKTracker(imnames)

lkt.detect_points()
lkt.draw()
for i in range(len(imnames)-1):
	lkt.track_points()
	lkt.draw()