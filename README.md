# Face Recognition
## Demo vedio
#### One person is in model and the other is not
Click photo below to watch demo vedio
[![Demo1](https://github.com/uuboyscy/Capture-Face/blob/master/sample1.png)](https://youtu.be/4p_tiyDbDA0 "Demo1")
#### Successfully recognize two persons who are in model
Click photo below to watch demo vedio
[![Demo2](https://github.com/uuboyscy/Capture-Face/blob/master/sample2.png)](https://youtu.be/9oSXastzVhI "Demo2")

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
    test_img.py (Recognize if there is any face on the photo)
    test_video.py (Real time vedio)
    test_captureFace.py (Merge the two programes abive)

#### Run on computer
    test_img.py (This can be run on computer)
    FaceModeling_v1.py (Cut the photoes -> Filter abnormal photoes -> Modeling)
    ConfusionMatrixAccuracy.py (Confusion matrix Accuracy)
