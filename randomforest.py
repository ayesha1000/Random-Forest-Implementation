
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#reading training dataset
dataset=pd.read_excel('TrainingSet.xlsx')
x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values

#training forest on training dataset with 100 trees
classify_100=RandomForestClassifier(n_estimators=100)
classify_100.fit(x,y)

#reading test dataset
testdata=pd.read_excel('TestSet1.xlsx')
testdata =testdata.drop(columns="plant")
testdata

#predicting plant value for test data 
test=testdata.iloc[:,0:4].values
test_100=classify_100.predict(test)
test_100

#storing result in testdata file
testdata['plant_100']=test_100
testdata

#training forest on training dataset with 300 trees
classify_300=RandomForestClassifier(n_estimators=300)
classify_300.fit(x,y)

#predicting plant value for test data 
test=testdata.iloc[:,0:4].values
test_300=classify_300.predict(test)
test_300

#storing result in testdata file
testdata['plant_300']=test_300
testdata
#training forest on training dataset with 500 trees
classify_500=RandomForestClassifier(n_estimators=500)
classify_500.fit(x,y)

#predicting plant value for test data 
test=testdata.iloc[:,0:4].values
test_500=classify_500.predict(test)
test_500

#storing result in testdata file
testdata['plant_500']=test_500
testdata


