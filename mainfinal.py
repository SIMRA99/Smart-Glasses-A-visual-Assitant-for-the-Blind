import face_recognition
from cv2 import cv2
import numpy as np
import os
import pyttsx3 
from googletrans import Translator     
# from pyscreenshot import ImageGrab
import time
import imutils
import argparse

from imutils.video import VideoStream

# from ultrasonic import read_distance


def read_distance():
    # Initialize the GPIO pins
    import RPi.GPIO as GPIO         # Import the Raspberry Pi GPIO library
    GPIO.setmode(GPIO.BCM)          # Use Broadcom GPIO pin numbering
    TRIG = 27                        # GPIO pin 02
    ECHO = 22                       # GPIO pin 03
    GPIO.setup(TRIG, GPIO.OUT)      # Set TRIG pin as output
    GPIO.setup(ECHO, GPIO.IN)       # Set ECHO pin as input
    GPIO.output(TRIG, GPIO.LOW)     # Initialize TRIG output as LOW

    # Send a HIGH signal to TRIG in order to trigger the sensor
    GPIO.output(TRIG, GPIO.HIGH)    # Send a HIGH pulse to TRIG
    time.sleep(0.00001)             # Wait 10 microseconds to trigger sensor
    GPIO.output(TRIG, GPIO.LOW)     # Set TRIG back to LOW

    # Once the sensor is triggered, it will send an ultrasonic pulse and set
    # the ECHO signal to HIGH. As soon as the receiver detects the original
    # ultrasonic pulse, the sensor will set ECHO back to LOW.

    # We need capture the duration between ECHO HIGH and LOW to measure how
    # long the ultrasonic pulse took on its round-trip.

    pulse_start = time.time()               # Record the pulse start time
    while GPIO.input(ECHO) != GPIO.HIGH:    # Continue updating the start time
        pulse_start = time.time()           # until ECHO HIGH is detected

    pulse_end = pulse_start                 # Record the pulse end time
    while time.time() < pulse_start + 0.1:  # Continue updating the end time
        if GPIO.input(ECHO) == GPIO.LOW:    # until ECHO LOW is detected
            pulse_end = time.time()
            break

    GPIO.cleanup()                  # Done with the GPIO, so let's clean it up

    # The difference (pulse_end - pulse_start) will tell us the duration that
    # the pulse travelled between the transmitter and receiver.
    pulse_duration = pulse_end - pulse_start

    # We know that sound moves through air at 343m/s or 34,300cm/s. We can now
    # use d=vÃ—t to calculate the distance. We need to divide by 2, since we only
    # want the one-way distance to the object, not the round-trip distance that
    # the pulse took.
    distance = 34300 * pulse_duration / 2

    # The sensor is not rated to measure distances over 4m (400cm), so if our
    # calculation results in a distance too large, let's ignore it.
    if distance <= 400:
        return distance
    else:
        return None






def say(text):
    os.system("echo '" + text + "' | festival --tts")
# from ultrasonic import read_distance

def obj_detection_face_recog():
    say('running object detection')
    print('running the code')
    thres = 0.45 # Threshold to detect object
    nms_threshold = 0.2

    classNames= []
    classFile = 'coco.names'
    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')


    print('here')
    #print(classNames)
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # image1 = face_recognition.load_image_file(os.path.abspath("/home/pi/Desktop/FR/Arif.jpg"))
    # image1_face_encoding = face_recognition.face_encodings(image1)[0]

    image2 = face_recognition.load_image_file(os.path.abspath("/home/pi/Desktop/FR/SIMRA1.jpg"))
    image2_face_encoding = face_recognition.face_encodings(image2)[0]
    print('105')
    image3 = face_recognition.load_image_file(os.path.abspath("/home/pi/Desktop/FR/hacheeba.jpg"))
    image3_face_encoding = face_recognition.face_encodings(image3)[0]

    image4 = face_recognition.load_image_file(os.path.abspath("/home/pi/Desktop/FR/SABA.jpg"))
    image4_face_encoding = face_recognition.face_encodings(image4)[0]

    known_face_encodings = [
    #image1_face_encoding,
        image2_face_encoding,
        image3_face_encoding,
        image4_face_encoding
    ]
    print('names')
    known_face_names = [
    #     "Arif",
        "SIMRA",
        "HASEEBA",
        "SABAA"
    ]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    print('130')


    #ap = argparse.ArgumentParser()
    #ap.add_argument("-p", "--picamera", type=int, default=-1,
    #	help="whether or not the Raspberry Pi camera should be used")
    #args = vars(ap.parse_args())
    # initialize the video stream and allow the cammera sensor to warmup

    print('init videostream')
    videostream = VideoStream(usePiCamera = True)
    print('starting')
    vs = videostream.start()
    print('started')

    time.sleep(2.0)

    nameobj = None


    print('loop')
    while True:
        frame = vs.read()
        classIds, confs, bbox = net.detect(frame,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
        #print(type(confs[0]))
        #print(confs)

        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        #print(indices)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(frame, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            cv2.putText(frame,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            

            nameobj=classNames[classIds[i][0]-1].upper()
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        print ("Face detected -- {}".format(face_names))
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        

        cv2.imshow("Output",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #espeak.synth('Hello')
        #espeak.synth(face_names)
        #espeak.synth(nameobj)
        #espeak.synth('Have a good day')
        if(len(face_names) == 1):
            os.system("echo '" + face_names[0] + "' | festival --tts")
            
        if nameobj is not None:
            distance = read_distance()
            text = nameobj + ' is ' + str(int(distance)) +'centimeters away'
            say(text)
            # os.system("echo '" + nameobj + "' | festival --tts")
    #     os.system("echo face_names | festival --tts")
    #     os.system("echo nameobj | festival --tts")
        #engine = pyttsx3.init()
        #engine.say('Hello')
        #engine.say(face_names)
        #engine.say(nameobj)
        #engine.say('Have a good day')
    #     engine.runAndWait()

    print('stopping stuff')
    cv2.destroyAllWindows()
    vs.stop()


        


