import argparse
import io
import re
import os

from google.cloud import storage
from google.cloud import vision
from google.protobuf import json_format
import cv2
import time
import random


user_id = random.randint(10000,100000)

print('User id: ', user_id)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'a.json'

def detect_faces(frame):
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image(content=frame.tobytes())

    response = client.face_detection(image=image)
    faces = response.face_annotations

    for face in faces:

        data = {'user_id' : user_id, 'anger' : face.anger_likelihood, 'joy' : face.joy_likelihood, 'surprise' : face.surprise_likelihood, 'sorrow' : face.sorrow_likelihood, 'confidence' : face.detection_confidence}
        print(data)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    retval, buffer = cv2.imencode('.jpg', frame)
    detect_faces(buffer)
    time.sleep(1)

cv2.destroyWindow("preview")
vc.release()

