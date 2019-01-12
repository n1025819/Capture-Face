# Face Recognition
## What you need
#### Install necessary packages
    What you will need :
    OpenCV
    opencv-contrib-python
    picamera
    matplotlib
    numpy
#### Model
    haarcascade_frontalface_default.xml
    It's a model of face provided by OpenCV  
    
## Programe
#### Run on RaspberryPi(Picam is needed)
    test_img.py (辨識出任意照片上有無人臉)
    test_video.py (即時影像串流)
    test_captureFace.py (即時影像串流並標示出人臉)

#### Run on computer
    test_img.py (這個也可以在電腦執行)
    FaceModeling_v1.py (切照片>篩照片>建模)
    ConfusionMatrixAccuracy.py (混淆矩陣算模型正確率)
