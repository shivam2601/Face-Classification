import cv2
import numpy as np
from keras.models import load_model
from image_commons import nparray_as_image, draw_with_alpha

import sys,os
size = 4

datasets = 'datasets'

(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (130, 100)

(images, lables) = [np.array(lis) for lis in [images, lables]]

model2 = cv2.createFisherFaceRecognizer()
model2.train(images, lables)




#droidcam android
rgb = cv2.VideoCapture(0)
#rgb = cv2.VideoCapture(0)

emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise','fear']


def _load_emoticons(emotions):
    
    return [nparray_as_image(cv2.imread('%s.png' % emotion, -1), mode=None) for emotion in emotions]


#emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise','fear']
emoticons = _load_emoticons(emotions)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX



model = load_model('face_reco.h5')
emo = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'happy', 4: 'sadness', 5: 'surprise', 6: 'fear'}

def get_emo(im):
    im = im[np.newaxis, np.newaxis, :, :]
    res = model.predict_classes(im,verbose=0)
    emo = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'happy', 4: 'sadness', 5: 'surprise', 6: 'fear'}
    return emo[res[0]]



def recognize_face(im):
    im = cv2.resize(im, (48, 48))
    return get_emo(im)


while True:
    ret, fr = rgb.read()
    flip_fr = cv2.flip(fr,1)
    if ret is True:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    else:
        continue
    faces = facec.detectMultiScale(gray, 1.3,5)
    
    for (x,y,w,h) in faces:
        fc = fr[y:y+h, x:x+w, :]
        gfc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        out = recognize_face(gfc)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        flip_fr = cv2.flip(fr,1)
        face2 = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face2, (width, height))
        prediction2 = model2.predict(face_resize)
        if prediction2[1]<500:

           cv2.putText(flip_fr,'%s - %.0f' % (names[prediction2[0]],prediction2[1]),(x-10, y-10), font,1,(0, 255, 0),2)
        else:
          cv2.putText(flip_fr,'not recognized',(x-10, y-10), font,1,(0, 255, 0),2)
        image_to_draw = emoticons[list(emo.keys())[list(emo.values()).index(out)]]

        draw_with_alpha(flip_fr, image_to_draw, (x, y, w, h))
        cv2.putText(flip_fr, out, (30, 30), font, 1, (255, 255, 0), 2)

    

    
    cv2.imshow('rgb', flip_fr)

    
    k = cv2.waitKey(1) & 0xEFFFFF
    if k==27:   
        break
    elif k==-1:
        continue
    else:
        # print k
        continue

rgb.release()
cv2.destroyAllWindows()