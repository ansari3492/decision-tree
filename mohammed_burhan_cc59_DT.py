# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:38:31 2018

@author: Lenovo
"""

import pandas as pd

dataset=pd.read_csv("tree_addhealth.csv")



for i in dataset:
    dataset[i] = dataset[i].fillna(dataset[i].mean())


    
    

    
features=dataset.drop("TREG1",axis=1)
labels=dataset["TREG1"]
#splitting the data into train and test
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)



#decision tree
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(features_train,labels_train)

labels_predict=regressor.predict(features_test)
score=regressor.score(features_test,labels_test)

#random forest tree amking
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(features_train,labels_train)
labels_predict2=rfr.predict(features_test)
score2=rfr.score(features_test,labels_test)

#prediction
result_predict1=regressor.predict([1,1,1,1,1,1,20,1,3,1,1,1,1,1,1,3,1,1,25.6,2.5,0,21.9,7,19])
result_predict2=rfr.predict([1,1,1,1,1,1,20,1,3,1,1,1,1,1,1,3,1,1,25.6,2.5,0,21.9,7,19])



#making confussion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_predict)















#visual understanding
features_grid=np.arange(min(features),max(features),0.1)
features_grid=features_grid.reshape(-1,1)
plt.scatter(features,labels,color='CMY')
plt.plot(features_grid,regressor.predict(features_grid),color = 'blue')
plt.title("age vs length(decision tree regression)")
plt.xlabel("age")
plt.ylabel("length")
plt.show()