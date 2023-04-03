from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2
import pyttsx3 
from googletrans import Translator  
from easyocr import Reader
import argparse
import time 

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\simra\Desktop\pytess\tesseract.exe'

cam = cv2.VideoCapture(0)
time.sleep(1.0)
# a=0

layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("Loading EAST text detector...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

ret,frame = cam.read()


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def recognize(im):
	
	orig = im.copy()
	(origH, origW) = im.shape[:2]
	(newW, newH) = (320, 320)
	rW = origW / float(newW)
	rH = origH / float(newH)
	im = cv2.resize(im, (newW, newH))
	(H, W) = im.shape[:2]
	
	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(im, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	results = list()
	for (startX, startY, endX, endY) in boxes:
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		im = orig[startY:endY, startX:endX]
	
		engine = pyttsx3.init()
		engine.say("text detected, Do I read?")
		engine.runAndWait()
		value=input("read?= ")
		# recognize=input("recognize?=")
		if value=="yes" :  
			a=0    
			img_name = "opencv{}.png".format(a)
			cv2.imwrite(img_name, frame)
			print("{} written!".format(img_name))
			a = a+1
		def cleanup_text(text):
		# 			# strip out non-ASCII text so we can draw the text on the image
		# 			# using OpenCV
				return "".join([c if ord(c) < 128 else "" for c in text]).strip()
		
					# construct the argument parser and parse the arguments
	# recognize=input("recognize?=")
	# if recognize=="yes":
		ap = argparse.ArgumentParser()
					# ap.add_argument("-i", "--image", required=True,
						# help="path to input image to be OCR'd")
		ap.add_argument("-l", "--langs", type=str, default="en",
					help="comma separated list of languages to OCR")
		ap.add_argument("-g", "--gpu", type=int, default=-1,
				help="whether or not GPU should be used")
		args = vars(ap.parse_args())

		# break the input languages into a comma separated list
		langs = args["langs"].split(",")
		print("[INFO] OCR'ing with the following languages: {}".format(langs))
		# load the input image from disk
		image = cv2.imread("ocr.jpg")
		# OCR the input image using EasyOCR
		print("[INFO] OCR'ing input image...")
		reader = Reader(langs, gpu=args["gpu"] > 0)
		results = reader.readtext(image)

		# loop over the results
		for (bbox, text, prob) in results:
			# display the OCR'd text and associated probability
			print("[INFO] {:.4f}: {}".format(prob, text))
			# unpack the bounding box
			(tl, tr, br, bl) = bbox
			tl = (int(tl[0]), int(tl[1]))
			tr = (int(tr[0]), int(tr[1]))
			br = (int(br[0]), int(br[1]))
			bl = (int(bl[0]), int(bl[1]))
			# cleanup the text and draw the box surrounding the text along
			# with the OCR'd text itself
			text = cleanup_text(text)
			cv2.rectangle(image, tl, br, (0, 255, 0), 2)
			cv2.putText(image, text, (tl[0], tl[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
			

			# recognize=input("recognize?=")
			# if recognize=="yes":
			engine = pyttsx3.init()
			engine.say(text)
			engine.runAndWait()
				# cv2.imshow("Image", image)
	

# ret,frame = cam.read()
# frame = recognize(frame)
# cv2.imshow("detection !",frame)
# cv2.destroyAllWindows
	


if (cam.isOpened()== False): 
  print("Error opening video stream or file")

while(cam.isOpened()):
  frame = recognize(frame)
  ret, frame = cam.read()
  if ret == True:

    cv2.imshow('Frame',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

cam.release()

cv2.destroyAllWindows()


