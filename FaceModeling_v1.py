# coding:utf-8
'''
1. Capture the face in each photo.
2. Save face photoes into ./dataset.
3. Maybe there are some mistakes while cutting photoes , so the improper photoes will be removed.
4. After removing , there maybe exist a few improper photoes that could not be detected , so the process will pause for 5 minutes so that you can remove those photoes manually.
5. Train model.
'''
import cv2
import os
import numpy as np
from PIL import Image
import time

## Cut photoes and save into /dataset/
if not os.path.isdir('./dataset'):
    os.mkdir('./dataset')

dtUserDir = os.listdir('./Face')
try:
    userId = int(os.listdir('./dataset')[len(os.listdir('./dataset')) - 1].split('.')[1]) + 1
except:
    userId = 1
    print('[INFO] There is no data in /dataset')
print('[INFO] New user ID will start from %s' % userId)

for user in range(userId - 1, len(dtUserDir)):
    dt = os.listdir('./Face/%s' % (dtUserDir[user]))
    count = 1
    for face in dt:
        # image path
        imagepath = r'./Face/%s/%s' % (dtUserDir[user], face)
        print(imagepath)

        # model
        face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

        # read image
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find face in image

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        print("[INFO] There are %s faces on the photo!" % (str(len(faces))))

        # The photo will not be modeled unless the photo has only one face in it
        if len(faces) != 1:
            continue

        for (x, y, w, h) in faces:
            # cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imwrite("./dataset/User." + str(userId) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        count += 1

    userId += 1

print('[INFO] Face photoes cut!')
print('[INFO] Removing improper photoes.')
time.sleep(3)

## Filter the photo -> remove improper photoes
datasetPath = r'./dataset'

for i in os.listdir(datasetPath):
    print(i)
    size = os.path.getsize(datasetPath + '/' + i)
    print(size)
    if size < 110000:
        os.remove(datasetPath + '/' + i)
        print('Deleted!')

n = len(os.listdir('./Face'))
print('[INFO] Abnormal photoes has been removed.')
print('[INFO] Second time checking...')

for i in range(1, n + 1):
    sizeUser = 0
    countUser = 0
    for j in os.listdir(datasetPath):
        sizeUser += int(os.path.getsize(datasetPath + '/' + j))
        countUser += 1
    avgSizeOfUser_i = sizeUser/countUser
    for j in os.listdir(datasetPath):
        print(j)
        size = os.path.getsize(datasetPath + '/' + j)
        print(size)
        if size < avgSizeOfUser_i*3/5:
            os.remove(datasetPath + '/' + j)
            print('Deleted!')

print('[INFO] Improper photoes removed.')
print('[INFO] Please check if there still exist improper photoes in your directory manually.')
print('[INFO] Model training will start 10 min later.')
time.sleep(600)

## Training
# Path for face image database
path = './dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        print('%s is modeling...'%imagePath)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
print('[INFO] Modling all data...')
# Save the model into trainer/trainer.yml
os.mkdir('./trainer')
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] %s faces trained. Exiting Program"%(len(np.unique(ids))))