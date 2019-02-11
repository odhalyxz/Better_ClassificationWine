#%matplotlib inline
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,matthews_corrcoef,f1_score,precision_score,recall_score
from sklearn.metrics import precision_recall_fscore_support


dict_classifiers = {

       "Nearest Neighbors": 
            {'classifier': KNeighborsClassifier(),
                 'params': [
                            {
                            'n_neighbors': [1, 3, 5, 7,9,11,13],
                            'metric': ['euclidean', 'cityblock','minkowski'],
                            'weights': ['uniform', 'distance']
                            }
                           ]
            },
            "Naive Bayes": 
            {'classifier': GaussianNB(),
                 'params': {}
            }
}

import time
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
num_classifiers = len(dict_classifiers.keys())



def batch_classify(X_train, y_train, X_test, y_test,namel_labels ):
    df_results = pd.DataFrame(
        data=np.zeros(shape=(num_classifiers,6)),
        columns = ['Classifier',
                    'Precision',
                    'MCC',
                    'Recall',
                    'F1-score',
                   'Training_time'])
    count = 0
    dataSet_Params = []
    for key, classifier in dict_classifiers.items():
        print("========================================================")
        print("Clasificador:",key)
        t_start = time.clock()
        
        gs_clf = GridSearchCV(classifier['classifier'], 
                      classifier['params'],
                      refit=True,
                        cv = 10, # 9+1
                        scoring = 'accuracy', # scoring metric
                        n_jobs = -1
                        )
                                                            
        # Use Train data to parameter selection in a Grid Search
        gs_clf = gs_clf.fit(X_train, y_train)
        # El mejor modelo
        model = gs_clf.best_estimator_
        # Los mejores parametros
        best_params = gs_clf.best_params_ 
        data = {}
        data['Num.']=count
        data['Clasificador'] = key
        data['Best_Params'] = best_params
        dataSet_Params.append(data)

        
        # Use best model and test data for final evaluation
        y_pred = model.predict(X_test)
        '''# Solo para verificar
        y_pred_train = model.predict(X_train)
        CR = classification_report(y_train,y_pred_train)
        print("------------------",CR)'''

        t_end = time.clock()
        t_diff = t_end - t_start

        CR = classification_report(y_test,y_pred)
        MC = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(MC,
                     index = namel_labels  , 
                     columns = namel_labels )
        print("Classification Report")
        print(cm_df)
        print("Matrix_Confusion")
        print(cm_df)

        cm_df.to_csv(str('save_Files/'+str(count+1))+'_'+key+'_Matrix_Confusion.csv')
        
        MCC = matthews_corrcoef(y_test, y_pred)
        F1 = (f1_score(y_test, y_pred, average="macro"))
        PS = (precision_score(y_test, y_pred, average="macro"))
        RS = (recall_score(y_test, y_pred, average="macro"))   
        #(precision,recall,fbeta_score,support )= precision_recall_fscore_support(y_test, y_pred,average='micro')




        df_results.loc[count,'Classifier'] = key
        df_results.loc[count,'Precision'] = PS
        df_results.loc[count,'MCC']= MCC
        df_results.loc[count,'Recall'] = RS
        df_results.loc[count,'F1-score'] =  F1
        df_results.loc[count,'Training_time'] = t_diff
        
        count+=1


    return df_results,dataSet_Params

#% ------------------


