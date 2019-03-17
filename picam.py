import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
#names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']
userlist = os.listdir('./userInfo')
names = []
for i in userlist:
    names.append(i.split('.')[1])

# Initialize and start realtime video capture
cam = cv2.VideoCapture(r'test8.MOV')
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Vedio writer object, encoding XVID
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output8.avi', fourcc, 20.0, (900, 600))

while True:

    ret, img = cam.read()
    img = cv2.flip(img, -1)  # Flip vertically

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('error')
        break

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 40):
            id = names[id]
            confidence = "  %s"%(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  %s"%(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    res = cv2.resize(img, (900, 600), interpolation=cv2.INTER_CUBIC)
    # Vedio writer
    out.write(res)
    # Show
    cv2.imshow('camera', res)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break


# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
out.release()
cv2.destroyAllWindows()