from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from app_logger.logger import APP_LOGGER
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model  import  LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_datasets as tfds

class Model_Finder:
    def __init__(self,file_object):
        self.file_object = file_object
        self.logger_object = APP_LOGGER()
        self.sv_classifier=SVC()
        self.naive_Bayes_classifier = make_pipeline(CountVectorizer(), MultinomialNB())
        self.xgb = XGBClassifier(objective='binary:logistic',n_jobs=-1)
    def get_best_params_for_svm(self,train_x,train_y):
        """
            Method Name: get_best_params_for_svm
            Description: get the parameters for the SVM Algorithm which give the best accuracy
            Output:  The model with the best parameters   
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            self.param_grid = {'tfidf__max_df': [0.7, 0.8, 0.9],  'tfidf__min_df': [0.1, 0.2, 0.3], 'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)], 'classifier__alpha': [0.1, 0.5, 1.0],  }
            self.grid = GridSearchCV(estimator=self.sv_classifier, param_grid=self.param_grid, cv=5,  verbose=3)
            self.grid.fit(train_x, train_y)
            self.kernel = self.grid.best_params_['kernel']
            self.C = self.grid.best_params_['C']
            self.random_state = self.grid.best_params_['random_state']
            self.sv_classifier = SVC(kernel=self.kernel,C=self.C,random_state=self.random_state)
            self.sv_classifier.fit(train_x, train_y)
            self.logger_object.log(self.file_object,'SVM best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svm method of the Model_Finder class')
            return self.sv_classifier
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'SVM training  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):
        """
            Method Name: get_best_params_for_xgboost
            Description: get the parameters for XGBoost Algorithm which give the best accuracy
            Output:  The model with the best parameters   
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            self.param_grid_xgboost = {"n_estimators": [100, 130], "criterion": ['gini', 'entropy'],"max_depth": range(8, 10, 1)}
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            self.grid.fit(train_x, train_y)
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.xgb = XGBClassifier(criterion=self.criterion, max_depth=self.max_depth,n_estimators= self.n_estimators, n_jobs=-1 )
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,'XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()
        

    def vectoring(self,traindata):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        #vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.8)
        X_train_tfidf = vectorizer.fit_transform(traindata)
        return vectorizer,X_train_tfidf

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
            Method Name: get_best_model
            Description: Find out the Model which has the best AUC score
            Output:  The best model name and the model object
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')
        try:
            vec,traindata=self.vectoring(train_x)
            X_test_tfidf = vec.transform(test_x)
            # svd = TruncatedSVD(5)
            # normalizer = Normalizer(copy=False)
            #logistic regression 
            logr_liblinear = LogisticRegression(max_iter=500)
           # logr_liblinear = logr_liblinear.fit(train_x.toarray(), train_y)
            logr_liblinear = logr_liblinear.fit(traindata, train_y)
           # logr_liblinear.feature_names=list(train_x.toarray().columns.values)
            logr_liblinear.feature_names = vec.get_feature_names_out()
            prediction_logistic = logr_liblinear.predict(X_test_tfidf)
            logistic_score = accuracy_score(test_y, prediction_logistic) * 100
            self.logger_object.log(self.file_object, 'Accuracy for Logistic:' + str(logistic_score)) 
            #navie bayes 
            naive_Bayes = GaussianNB()
            naive_Bayes =naive_Bayes.fit(traindata.toarray(), train_y)
            prediction_naive_Bayes = naive_Bayes.predict(X_test_tfidf.toarray())
            naive_Bayes_score = accuracy_score(test_y, prediction_naive_Bayes) * 100
            self.logger_object.log(self.file_object, 'Accuracy for naive Bayes :' + str(naive_Bayes_score)) 
             #decission tree
            decision_tree = DecisionTreeClassifier()
            decision_tree=decision_tree.fit(traindata,train_y)
            prediction_destree= decision_tree.predict(X_test_tfidf)
            desctree_score = accuracy_score(test_y, prediction_destree) * 100
            self.logger_object.log(self.file_object, 'Accuracy for Descision Tree:' + str(desctree_score)) 
            # #svm model
            # svm_clf = SVC()
            # svm_clf = svm_clf.fit(train_x.toarray(), train_y)
            # svm_pred = svm_clf.predict(train_x.toarray())
            # svm_score = accuracy_score(test_y, svm_pred) * 100
            # self.logger_object.log(self.file_object, 'Accuracy for naive Bayes :' + str(svm_score)) 
            all_models ={"Naive Bayes":naive_Bayes,"Logistic":logr_liblinear,"decission Tree":decision_tree}
            all_models_scores ={"Naive Bayes":naive_Bayes_score,"Logistic":logistic_score,"decission Tree":desctree_score}
           # best_model_score=max(all_models_scores.values())
            best_model_name=max(all_models_scores, key=all_models_scores.get)
            best_model=all_models[best_model_name]
            return best_model_name,best_model         
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()



    def get_best_params_for_navieBayes(self,train_x,train_y):
        """
            Method Name: get_best_params_for_navieBayes
            Description: get the parameters for the Navie Bayes Algorithm which give the best accuracy
            Output:  The model with the best parameters   
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_navieBayes method of the Model_Finder class')
        try:
            self.param_grid = {'countvectorizer__ngram_range': [(1, 1), (1, 2)],  'multinomialnb__alpha': [0.1, 0.5, 1.0]}
            self.grid = GridSearchCV(estimator=self.naive_Bayes_classifier, param_grid=self.param_grid, cv=5,  verbose=3)
            self.grid.fit(train_x, train_y)
            self.kernel = self.grid.best_params_['kernel']
            self.C = self.grid.best_params_['C']
            self.random_state = self.grid.best_params_['random_state']
            self.naive_Bayes_classifier = MultinomialNB(kernel=self.kernel,C=self.C,random_state=self.random_state)
            self.naive_Bayes_classifier.fit(train_x, train_y)
            print("Best hyperparameters:", self.grid.best_params_)
            # accuracy = grid_search.score(test_data, test_labels)
            # print(f"Test accuracy: {accuracy:.4f}")
            self.logger_object.log(self.file_object,'Navie Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_navieBayes method of the Model_Finder class')
            return self.naive_Bayes_classifier
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_best_params_for_navieBayes method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Navie bayes training  failed. Exited the get_best_params_for_navieBayes method of the Model_Finder class')
            raise Exception() 