import numpy as np
import pandas as pd
import cv2 as cv2

def drop_rows_by_value(df,value,columns=None):
  if(columns is None):
    columns=list(df.columns)
    
  for i in columns:  
    df=df[df[i]!=value]
  return df

def count_df_rows(df, col_names):        
    num_rows = df.shape[0]    
    labels = df[col_names].values    
    counts = np.sum(labels, axis=0)
    class_num = dict(zip(col_names, counts))
    return num_rows, class_num



def preprocess_image(path):        
    img = cv2.imread(path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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
    #crop image
    crop_img = resized_img[0:320, 0:320]
    return crop_img

