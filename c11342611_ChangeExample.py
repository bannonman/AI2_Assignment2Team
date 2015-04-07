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
#print(set(ids[:]))
# read in trainingset data and adding headers : not sure if this is the way to assign the headers as i have hardcoded 1,2,3,4,5,6.... to stop out_of_index error
columnHeadings=['Id','Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area','1','2','3','Soil_Type','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','Cover_Type']
Tdata = pd.read_csv("data/trainingset.txt",header=None,names=columnHeadings,index_col=False,na_values=['?'],nrows=435148)
# Extract Target Feature
targetLabels = Tdata['Cover_Type']
# Extract Numeric Descriptive Features
numeric_features = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
numeric_dfs = Tdata[numeric_features]
# Extract none-Categorical Descriptive Features
cat_dfs = Tdata.drop(numeric_features + ['Cover_Type'],axis=1)

cat_dfs= numeric_dfs.T.to_dict().values()
#convert to numeric encoding
vectorizer = DictVectorizer( sparse = False )
vec_cat_dfs = vectorizer.fit_transform(cat_dfs) 
# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs ))


decTreeModel = tree.DecisionTreeClassifier(criterion='entropy')
#Algorithm used, no idea what this does
#clf = svm.SVC(gamma=0.001, C=100)

#Again read above
decTreeModel.fit(train_dfs,targetLabels)
###############################################

Qdata = pd.read_csv("data/queries.txt",header=None,names=columnHeadings,index_col=False,na_values=['?'],nrows=145864)

numeric_features = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
numeric_dfs = Qdata[numeric_features]

# Extract none-Categorical Descriptive Features
cat_dfs = Qdata.drop(numeric_features + ['Cover_Type'],axis=1)

cat_dfs= numeric_dfs.T.to_dict().values()
#print(cat_dfs)

ids = Qdata['Id']

#q = {'Elevation':[2000,2516],'Aspect':[23,23],'Slope':[6,6],'Horizontal_Distance_To_Hydrology':[150,150],'Vertical_Distance_To_Hydrology':[4,4],'Horizontal_Distance_To_Roadways':[658,658],'Hillshade_9am':[216,216],'Hillshade_Noon':[227,227],'Hillshade_3pm':[147,147],'Horizontal_Distance_To_Fire_Points':[5541,5541]}
col_names = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
qdf = pd.DataFrame.from_dict(list(cat_dfs),orient="columns")
#extract the numeric features
q_num = qdf[numeric_features].as_matrix() 
#convert the categorical features
q_cat = qdf.drop(numeric_features,axis=1)
#print(qdf)
q_cat_dfs = q_cat.T.to_dict().values()
q_vec_dfs = vectorizer.transform(q_cat_dfs) 
#merge the numeric and categorical features
query = np.hstack((q_num, q_vec_dfs ))
#Use the model to make predictions for the 2 queries
predictions = decTreeModel.predict(query[:])
print("Predictions!")
print("------------------------------")
print(set(ids[:]))
#f = open("data/predictions.txt", 'w')

#for num in range(0,145864):
   # f.write(ids[num]+','+predictions[num]+'\n')


#print(ids[:]+':'+predictions) 

