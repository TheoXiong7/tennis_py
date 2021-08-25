import cv2
import numpy as np
import math
import time
from typing import NamedTuple

hsv_lower = np.array([20, 100, 95])
hsv_upper = np.array([90, 255, 255])

#hsv_lower = np.array([20, 50, 180])
#hsv_upper = np.array([90, 170, 255])

def analyze_video(video_path):
	cap = cv2.VideoCapture(video_path)
	prev_frame_time = 0
	new_frame_time = 0

	while True:
		# time.sleep(.5)
		ret, frame = cap.read()

		detect = find_circles(frame)
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = str(int(fps))
		cv2.putText(detect, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (252, 186, 3), 2)

		cv2.imshow('tennis', detect)
		if chr(cv2.waitKey(1) & 255) == 'q':
			break

def analyze_camera():
	"""
	imgname = 'r7.jpeg'
	img = cv2.imread('input\\' + imgname)
	img = find_circles(img)
	cv2.imshow('tennis', img)
	cv2.imwrite('output\\' + imgname, img)
	cv2.waitKey(0)

	"""
	cap = cv2.VideoCapture(0)
	prev_frame_time = 0
	new_frame_time = 0

	while True:
		# time.sleep(.5)
		ret, frame = cap.read()

		detect = find_circles(frame)
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = str(int(fps))
		cv2.putText(detect, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (252, 186, 3), 2)

		cv2.imshow('tennis', detect)
		if chr(cv2.waitKey(1) & 255) == 'q':
			break
	

def draw_circles(img, circles):
	cimg = img.copy()
	if type(circles) == type(None):
		return cimg
	circles = np.round(circles[0, :]).astype("int")
	#print(circles)
	for (x, y, r) in circles:
		cv2.circle(cimg, (x, y), r, (0, 255, 0), 2)
	return cimg

def find_circles(image):

	ratio = image.shape[1] / image.shape[0]
	h, w = 700, int(700 * ratio)
	image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
	original = image.copy()

	blur = cv2.GaussianBlur(image, (11, 11), 0)

	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	#return mask
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		# only proceed if the radius meets a minimum size
		if radius > 0:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(original, (int(x), int(y)), int(radius), (0, 0, 255), 2)
	return original
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
	gray_lap = cv2.Laplacian(gray_blur, cv2.CV_8UC1, ksize=5)
	dilate_lap = cv2.dilate(gray_lap, (3, 3))
	lap_blur = cv2.bilateralFilter(dilate_lap, 5, 9, 9)

	#circles = cv2.HoughCircles(lap_blur, cv2.HOUGH_GRADIENT, 1, 100, param1 = 50, param2 = 30, minRadius = 60, maxRadius = 0)
	circles = cv2.HoughCircles(lap_blur, cv2.HOUGH_GRADIENT, 16, 200, param2=450, minRadius=60, maxRadius=0)
	cimg = draw_circles(original, circles)
	return cimg
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hsv = cv2.inRange(image, hsv_lower, hsv_upper)
	mask = cv2.GaussianBlur(hsv, (15, 15), 0)
	return mask
	gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	return gray
	# print(mask.shape)
	# img = mask.clone()
	# h = img.rows
	# w = img.cols

	"""
	# Find contours
	cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Extract contours depending on OpenCV version
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]

	# Iterate through contours and filter by the number of vertices
	for c in cnts:
		perimeter = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
		if len(approx) > 5:
			cv2.drawContours(original, [c], -1, (255, 80, 80), -1)
			print('ball found')
		else:
			print('ball not found')
		
	
	"""

if __name__ == '__main__':
    #analyze_video('input\\tennis.mp4')
	analyze_camera()