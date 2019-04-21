from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
import numpy as np

class KNN:
    knn=KNeighborsClassifier()
    
    def __init__(self):
        #steps=[('scaler',StandardScaler()),('KNN',self.knn)]
        #self.pipeline=Pipeline(steps)
       param_grid={'n_neighbors':np.arange(1,50)}
       self.grid_search=GridSearchCV(self.knn, param_grid, cv=5)      
    
    
    #def apply_fit_predict(self,df,labels,test):
       #self.pipeline.fit(df.astype(float),labels)
       #return self.pipeline.predict(test.astype(float))
        
    
    
    
    def apply_grid_search(self,train,labels):
      self.grid_search.fit(train, labels)
      return self.grid_search.best_params_
   
    
    def apply_fit_predict(self,train,labels,test,param):
        self.knn=KNeighborsClassifier(param['n_neighbors'])
        print(param['n_neighbors'])
        self.knn.fit(train,labels)
        return self.knn.predict(test)
   
       
    def print_accuracy(self,y_test,y_pred):
        print("accuracy: "+str(accuracy_score(y_test,y_pred)))
