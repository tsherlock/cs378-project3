import lktrack

imnames = []

path = "test_data/ball_4_frames/"
i = 0
while i <= 326:
	s = str(i)
	while len(s) < 3:
		s = "0" + s
	imnames.append(path + "frame_" + s + ".png")
	i = i + 1

#print imnames
lkt = lktrack.LKTracker(imnames)

lkt.detect_points()
lkt.draw()
for i in range(len(imnames)-1):
	lkt.track_points()
	lkt.draw()
