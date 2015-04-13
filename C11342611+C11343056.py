#"Corey Bannon + John Kiernan"
#"C11342611 + C11343056"
#"DT228 / 4"
#"10 - 04 - 2015"
#"corey.bannon@mydit.ie"
#"Assignment 2 AI2"
__author__ = "Corey Bannon + John Kiernan"
__studentID__ = "C11342611 + C11343056"
__courseyear__ = "DT228 / 4"
__date__ = "10 - 04 - 2015"
__email__ = "corey.bannon@mydit.ie"
__status__ = "Assignment 2 AI2"


from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime

print("Starting......")
print(datetime.datetime.now())
       
def main():
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

       clf = RandomForestClassifier(n_jobs=1)
       clf.fit(train_dfs, targetLabels) 
       #decTreeModel = tree.DecisionTreeClassifier(criterion='entropy')
       #Algorithm used
       #decTreeModel.fit(train_dfs,targetLabels)

       columnHeadings=['Id','Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area','1','2','3','Soil_Type','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','Cover_Type']
       Qdata = pd.read_csv("data/queries.txt",header=None,names=columnHeadings,index_col=False,na_values=['?'],nrows=145864)
       numeric_features = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
       numeric_dfs = Qdata[numeric_features]
       # Extract none-Categorical Descriptive Features
       cat_dfs = Qdata.drop(numeric_features + ['Cover_Type'],axis=1)
       cat_dfs= numeric_dfs.T.to_dict().values()
       ids = Qdata['Id']
       #q = {'Elevation':[2000,2516],'Aspect':[23,23],'Slope':[6,6],'Horizontal_Distance_To_Hydrology':[150,150],'Vertical_Distance_To_Hydrology':[4,4],'Horizontal_Distance_To_Roadways':[658,658],'Hillshade_9am':[216,216],'Hillshade_Noon':[227,227],'Hillshade_3pm':[147,147],'Horizontal_Distance_To_Fire_Points':[5541,5541]}
       qdf = pd.DataFrame.from_dict(list(cat_dfs),orient="columns")
       #extract the numeric features
       q_num = qdf[numeric_features].as_matrix() 
       #convert the categorical features
       q_cat = qdf.drop(numeric_features,axis=1)
       q_cat_dfs = q_cat.T.to_dict().values()
       q_vec_dfs = vectorizer.transform(q_cat_dfs) 
       #merge the numeric and categorical features
       query = np.hstack((q_num, q_vec_dfs ))
       #Use the model to make predictions for the 2 queries
       predictions = clf.predict(query[:])

       
       print("Writing predictions to file......")
       print(datetime.datetime.now())
       
       f = open("solutions/c11342611+C11343056.txt", 'w')
       for num in range(0,145864):
              f.write(ids[num]+','+predictions[num]+'\n')
       f.close()
       
       print("Report saved")
       print(datetime.datetime.now())
       
       #Split the data: 60% training : 40% test set
       instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.74, random_state=0)

       #run a 10 fold cross validation on this model using the full census data
       scores=cross_validation.cross_val_score(clf, instances_train, target_train, cv=10)
       #the cross validaton function returns an accuracy score for each fold
       print("Entropy based Model:")
       print("Score by fold: " + str(scores))
       #we can output the mean accuracy score and standard deviation as follows:
       print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
       print("\n\n")
       
#If it's to be used as module in other python script the main won't execute
if __name__ == "__main__":
    main()
