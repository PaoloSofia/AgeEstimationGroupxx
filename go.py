import os
import tensorflow as tf 
import numpy as np 
import csv 
import pandas as pd
import dlib
import cv2 as cv
from random import getrandbits,randint
from imutils import rotate

EXT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = os.path.abspath(os.path.join(EXT_ROOT, os.pardir))


modelFile = os.path.join(EXT_ROOT,'data','res10_300x300_ssd_iter_140000_fp16.caffemodel')
configFile = os.path.join(EXT_ROOT,'data',"deploy.prototxt")


#------------ DEFINIZIONE DELLE FUNZIONI DI PREPROCESSING ----------------

class FaceDetector:
    net = None
    def __init__(self, min_confidence=0.5):
        print ("FaceDetector -> init")
        self.net = cv.dnn.readNetFromCaffe(configFile, modelFile)
        self.min_confidence = min_confidence
        print ("FaceDetector -> init ok")
    
    def detect(self, image):
        blob = cv.dnn.blobFromImage(image, 1.0, (100, 100), [106, 121, 150], False, False)
        frameHeight, frameWidth, channels = image.shape
        self.net.setInput(blob)
        detections = self.net.forward()
        faces_result=[]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_confidence:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                f = (x1,y1, x2-x1, y2-y1)
                if f[2]>1 and f[3]>1:
                    faces_result.append({
                        'roi': f,
                        'type': 'face',
                        'img': image[f[1]:f[1]+f[3], f[0]:f[0]+f[2]],
                        'confidence' : confidence
                    })
        return faces_result
    
    def __del__(self):
        print ("FaceDetector -> bye")



def preprocessing(img,resize=False,size=None):
  """Effettua il preprocessing dell'immagine

  Parameters
  ----------
  img : np.ndarray
      L'immagine da preprocessare
  label: int
      Label associata all'immagine
  resize: bool, optional
      Se true effettua il resize dell'immagine 
  size: tuple, optional
      Se resize = True, effettua il resize della dimensione specificata

  Returns
  -------
  tf.Tensor
      L'immagine preprocessata sotto forma di tensore
  """
  if resize:
    img = cv.resize(img,(size,size))
  
  #applica il clahe
  lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
  lab_planes = cv.split(lab)
  lab_planes[0] = clahe.apply(lab_planes[0])
  lab = cv.merge(lab_planes)
  img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)


  img = img.astype(np.float32)
  img = img / 255.0
 
  return img


def findRelevantFace(objs, W,H):
  """Trova il volto principale tra i volti trovati nella face detection
  Parameters
  ----------
  objs : tf.Tensor
        lista di dizionari, ogni elemento della lista contiene informazioni su un volto rilevato
  W : tf.Tensor
        larghezza dell'immagine originale
  H : int
        lunghezza dell'immagine originale
  Returns
  bool
    True se label != 255, False altrimenti

  """
  mindistcenter = None
  minobj = None
  for o in objs:
      cx = o['roi'][0] + (o['roi'][2]/2)
      cy = o['roi'][1] + (o['roi'][3]/2)
      distcenter = (cx-(W/2))**2 + (cy-(H/2))**2
      if mindistcenter is None or distcenter < mindistcenter:
          mindistcenter = distcenter
          minobj = o
  return minobj


def detect_and_align(filename,size):
  """Effettua la detection della faccia nell'immagine e allinea il volto.

  Parameters
  ----------
  img : tf.Tensor
        Tensore che contiene l'immagine
  label : tf.Tensor
        Tensore che contine la label
  size : tuple
        Dimensione desiderata dell'immagine preprocessata
  Returns
  Tf.Tensor
    Tensore che contine l'immagine preprocessata
  """
  resize = True

  img_array = cv.imread(os.path.join(ROOT_PATH,'vggface2_test/test',filename))
  det_imgs = detector.detect(img_array)
  if len(det_imgs) > 0:
    rel = findRelevantFace(det_imgs,img_array.shape[0],img_array.shape[1])
    roi = rel['roi']
    rect = dlib.rectangle(roi[0],roi[1],roi[0]+roi[2],roi[1]+roi[3])
    face = sp(img_array, rect)
    img_array = dlib.get_face_chip(img_array, face, size=size[0])

  return preprocessing(img_array,True,size[0])

  
#------------ CARICA MODELLO DA FILE ----------------

tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()
model = tf.keras.models.load_model(os.path.join(EXT_ROOT,'model','model.h5'),
                                compile=False)


# ------------ PREDIZIONE SU DATI DI TEST -----------
detector=FaceDetector()
clahe = cv.createCLAHE(2,(3,3))
predictor_path = os.path.join(EXT_ROOT,"data","shape_predictor_68_face_landmarks.dat")
sp = dlib.shape_predictor(predictor_path)
net = cv.dnn.readNetFromCaffe(configFile, modelFile)
size = (100,100,3)



df = pd.DataFrame(pd.read_excel(
    os.path.join(EXT_ROOT,'data','test_reduced.xlsx'),
    header=None,
    engine='openpyxl'
    ))


predictions = []
for column_name, item in df.iteritems():
    for row in item.iteritems():
        start_idx = row[1].index('n')
        stop_idx = row[1].index('.jpg') + 4
        filename = row[1][start_idx:stop_idx]
        img = detect_and_align(filename,size)
        pred = np.argmax(model.predict(img.reshape(1,size[0],size[1],size[2])))+1
        predictions.append([filename,pred])


with open(os.path.join(EXT_ROOT,'data','GROUP_22.csv'), 'w') as f:       
    write = csv.writer(f) 
    write.writerows(predictions) 