from pandas import DataFrame
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time 

# read in trainingset data and adding headers : not sure if this is the way to assign the headers as i have hardcoded 1,2,3,4,5,6.... to stop out_of_index error
columnHeadings=['Id','Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area','1','2','3','Soil_Type','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','Cover_Type']
Tdata = pd.read_csv("data/testtrainingset.txt",header=None,names=columnHeadings,index_col=False,na_values=['?'],nrows=24)
# Extract Target Feature
targetLabels = Tdata['Cover_Type']
# Extract Numeric Descriptive Features
numeric_features = ['Id','Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
numeric_dfs = Tdata[numeric_features]
# Extract none-Categorical Descriptive Features
cat_dfs = Tdata.drop(numeric_features + ['Cover_Type'],axis=1)

cat_dfs = cat_dfs.to_dict()
print(cat_dfs[0])
time.sleep(15)


#convert to numeric encoding
vectorizer = DictVectorizer( sparse = False )
vec_cat_dfs = vectorizer.fit_transform(cat_dfs) 
# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), cat_dfs ))
print('Prediction:',train_dfs)