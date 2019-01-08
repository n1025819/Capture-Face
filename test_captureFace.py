# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# face model
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # capture face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.15,
                                          minNeighbors=5,
                                          minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # draw on face
    for (x, y, w, h) in faces:
        # cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)
        cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break