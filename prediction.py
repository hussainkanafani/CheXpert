import numpy as np
import pandas as pd
import sys
import csv
from tensorflow import keras
import h5py
import cv2
from PIL import Image
        
def predict_all(model, studies, result_csv_path='results.csv'):
    results = [['Study', 'Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion']]
    for study in studies:
        studyname = study[0]
        studyresults = predict_study(model,study[1])
        studyresults = [str(result) for result in studyresults]
        results.append([studyname] + studyresults)
    with open(result_csv_path, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(results)
    csvFile.close()
                
def predict_study(model, images):
    scores = model.predict(images)
    all_scores = [scores[i] for i in range(len(scores))]
    max_scores = np.maximum.reduce(all_scores)
    return max_scores

def load_studies(csv_path,img_size=320):
    img_paths = pd.read_csv(csv_path, usecols=['Path'])
    studies = []
    current_patient = ''
    current_images = []
    current_study = ''
    for img_row in img_paths.values:
        img_path = img_row[0]
        parts = img_path.split('/')
        if len(parts) < 5 or parts[4] == '':
            continue
        patient = parts[2]
        study = '/'.join(parts[0:4])
        if patient != current_patient:
            if len(current_study) > 0 and len(current_images) > 0:
                studies.append(current_images)
            current_patient = patient
            current_study = study
            current_images = [study]
        elif study != current_study:
            if len(current_study) > 0 and len(current_images) > 0:
                studies.append(current_images)
            current_study = study
            current_images = [study]
        image = load_image(img_path,img_size)
        current_images.append(image)
    if len(current_images) > 0:
        studies.append(current_images)
    return studies

def load_image(image_path,img_size=320):
    img = cv2.imread(image_path, 1) 
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    if max_ind == 1:
        # width fixed at 320
        wpercent = (320 / float(size[0]))
        hsize = int((size[1] * wpercent))
        new_size = (hsize, 320)

    else:
        # height fixed at 320
        hpercent = (320 / float(size[1]))
        wsize = int((size[0] * hpercent))
        new_size = (320, wsize)
    resized_img = cv2.resize(img, new_size)
    resized_img = resized_img[0:img_size, 0:img_size]
    resized_img = standardize_images(resized_img)
    return np.array(resized_img).reshape(-1,img_size,img_size, 3)

def standardize_images(img_batch):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img_batch = (img_batch - imagenet_mean) / imagenet_std
    return img_batch

if __name__ == "__main__":
    args = sys.argv[1:]
    if (len(args) < 2):
        print("missing arguments")
    else:
        img_size = 320
        studies = load_studies(args[0],img_size)
        model = keras.models.load_model('./src/model.h5')
        predict_all(model,studies,args[1])

