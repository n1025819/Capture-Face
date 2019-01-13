# coding:utf-8
import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
print('Loading model file...')
recognizer.read('./trainer/trainer.yml')
# cascadePath = "haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)

# model
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

#
os.mkdir('./tester/tptn')
tptn = 0
all = 0
checkId = 0

# names related to ids: example ==> JieCheng-Pan: id=1,  etc
'''names = ['None', 'JieCheng-Pan', 'YiYing-Chen', 'MinQi-Qiu', 'ShengZhi-Qiu', 'ChengYou-Shi', 'YoHeng-Guo', 'YoQing-Chen', 'LvAn-Lin',
         'JunLong-chen', 'YuZHe-Qiu']'''
userlist = os.listdir('./userInfo')
names = []
for i in userlist:
    names.append(i.split('.')[1])

testerPath = r'./tester'
for user in os.listdir(testerPath):
    print('user ', user)
    imgCount = 0
    for img in os.listdir(testerPath + '/' + user):
        # image path
        imagepath = testerPath + '/' + user + '/' + img
        print(imagepath)

        # read image
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # fine face in image
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor = 1.2,
                                              minNeighbors = 5,
                                              minSize = (6,6))

        print("There are %s faces on the photo!"%(str(len(faces))))
        if int(len(faces)) > 1 or int(len(faces)) == 0:
            continue

        for(x,y,w,h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            checkId = id # used to check if int(checkId) == int(user)

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  %s" % (round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  %s" % (round(100 - confidence))

            cv2.putText(image, str(id), (x + 5, y - 5), font, 3, (255, 255, 255), 5)
            cv2.putText(image, str(confidence), (x + 5, y + h - 5), font, 3, (255, 255, 0), 5)
            # if TP/TN -> save the photo
            try:
                if int(checkId) == int(user):
                    cv2.imwrite('./tester/tptn/%s.%s.%s.jpg'%(id, imgCount, confidence), image)
                    tptn += 1
            except:
                print('stop!')

            imgCount += 1

        print('id=',id)
        print('confidence=',confidence)
        print('-')
        all += 1
        res = cv2.resize(image, (600, 800), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Find Faces!",res)
        cv2.waitKey(1)

print('\nTP and TN : %s'%tptn)
print('Data amount : %s'%all)
print('ACC : %s'%(tptn/all))