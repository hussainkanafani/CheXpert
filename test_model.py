import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
from itertools import cycle
from skimage.transform import resize
from model import ModelFactory
import pandas as pd
import cv2
import util
import sys
import getopt

csv_valid_path="CheXpert-v1.0-small/valid.csv"
chexpert_targets= ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
testing_labels= ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']


def read_test_data(csv_path,class_names):
  images=[]
  labels=[]
  df=pd.read_csv(csv_path)
  for i,row in df.iterrows():
    path = row["Path"] 
    img = util.preprocess_image(path)    
    images.append(img)
    labels.append(row[class_names])
  return np.asarray(images), np.asarray(labels)



def plot_ROC(v_labels_df, predictions_df):
  all_scores=[]
  plt.figure()
  colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green'])
  for label,color in zip(v_labels_df.columns,colors):
    #calculate roc_auc score
    score= roc_auc_score(v_labels_df[label].astype(int), predictions_df[label])    
    all_scores=np.append(all_scores,score)
    #plot roc curve
    fpr, tpr, thresholds = metrics.roc_curve(
    v_labels_df[label].astype(int), predictions_df[label], pos_label=1)
    plt.plot(fpr, tpr, color, label=label+' = {:0.4f}'.format(score))
  #calc mean AUC and plot
  title =' Mean AUC = {:0.4f}'.format(np.mean(all_scores))                 
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.title(title)    
  plt.show()
  
    

  
def test_model(weights_path, model_name):

    v_data, v_labels = read_test_data(csv_valid_path,chexpert_targets)
    image_dimension = 320
    
    model_factory = ModelFactory()
    model = model_factory.get_model(
      chexpert_targets,
      model_name=model_name,
      weights_path=weights_path,
      input_shape=(image_dimension, image_dimension, 3))       
    predictions = model.predict(v_data)
    
    if  len(predictions) and  len(v_labels):
      
      predictions_df=pd.DataFrame(data=predictions,columns=chexpert_targets)
      v_labels_df=pd.DataFrame(data=v_labels,columns=chexpert_targets)  
      predictions_df=predictions_df[testing_labels]
      v_labels_df=v_labels_df[testing_labels]

      plot_ROC(v_labels_df,predictions_df)
      
    return predictions,v_labels  

if __name__ == '__main__':
    argv = sys.argv[1:]
    weights_path = None
    model_name = 'DenseNet121'
    try:
        opts, args = getopt.getopt(argv,"w:m:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-m","--modelName"):  
            model_name = arg
        elif opt in ("-w","--weightsPath"):
            weights_path = arg		
    test_model(weights_path, model_name)