import pandas 
from File_Operations.fileMethods import File_Operation
from Data_Preprocessing.preprocessing import Preprocesser
from app_logger.logger import APP_LOGGER
from Prediction_Data_Ingestion.predDataLoader import Data_Getter
from Prediction_Raw_Data_Validations.predRawValidations import RawValidations
from ArticleText_Extraction.ExtractArticleText import  extract_article_text
from Prediction_Model.keywords_predictFromModel import keywords_prediction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from Best_Model_Finder.tuner import Model_Finder
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack

class prediction:
    def __init__(self,path):
        self.file_object = open("Log_Files_Collection/Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log = APP_LOGGER()
        if path is not None:
            self.pred_data_val = RawValidations(path)
    
    def predictionFromModel(self):
        try:
            self.pred_data_val.deletePredictionFile()
            self.log.log(self.file_object,'Start of Prediction')
            data_getter=Data_Getter(self.file_object)
            data=data_getter.get_data()
            print(type(data))
            preprocessor=Preprocesser()
            # is_zerona_present,cols_with_nazerovalues=preprocessor.getColumns_with_NAZero(data)
            # if(is_zerona_present):
            #     data=preprocessor.remove_Columns(data,cols_with_nazerovalues)
            # data = preprocessor.replaceInvalidValuesWithNull(data)
            # is_null_present,cols_with_missing_values=preprocessor.isNull_Present(data)
            # if(is_null_present):
            #     data=preprocessor.impute_missing_values(data,cols_with_missing_values)
            #data = preprocessor.encodeCategoricalValues(data)
          #  data=preprocessor.dropColumns_with_Constant(data)
          #  data=preprocessor.drop_Features_with_Coorelation(data)
          #  data=preprocessor.remove_outliers(data)
            X,Y=preprocessor.separate_label_feature(data,label_column_name='ArticleCategory')
            # X_train = np.asarray(x_train)
            # Y_train = np.asarray(y_train)
            # X_test = np.asarray(x_test)
            # Y_test = np.asarray(y_test)
       #     X=preprocessor.remove_Columns(X,['qty_and_directory','qty_equal_params'])
           # cols_to_drop=preprocessor.get_columns_with_zero_std_deviation(X)
           # X=preprocessor.remove_Columns(X,cols_to_drop)
            file_loader=File_Operation(self.file_object)
            Logistic=file_loader.load_model('Logistic')
            f_names=Logistic.feature_names
            #self.log.log(self.file_object,f'The column values for the prediction data for Logistic  is {f_names}')
            #get_traincoldata=X[f_names]
            #clusters=Logistic.predict(get_traincoldata)
            #get_traincoldata['ArticleCategory']=Y
            # get_traincoldata['cluster']=clusters
            # clusters=get_traincoldata['cluster'].unique()
            # for i in clusters:
            #     cluster_data= get_traincoldata[get_traincoldata['cluster']==i]
            # get_traincoldata = np.asarray(X['ArticleText'])
            # get_traincoldata['ArticleCategory']=Y
            Article_cat = list(Y)
            # x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
            # X_train = np.asarray(x_train)
            # Y_train = np.asarray(y_train)
            # X_test = np.asarray(x_test)
            # Y_test = np.asarray(y_test)
            model_finder=Model_Finder(self.file_object)
            vec,traindata=model_finder.vectoring(X)
            new_data_tfidf =vec.transform(X)
            print(new_data_tfidf.shape)
            #    cluster_data=cluster_data.drop(labels=['ArticleCategory','cluster'],axis=1)
            #    self.log.log(self.file_object,f'The column values for the prediction data is {cluster_data.columns}')
              #  cluster_data = cluster_data.drop(['Cluster'],axis=1)
            model_name = 'Logistic'
            model = file_loader.load_model(model_name)
            #model_data=get_traincoldata.drop(labels=['ArticleCategory'],axis=1)
            result=list(model.predict(new_data_tfidf))
            result = pandas.DataFrame(list(zip(Article_cat,result)),columns=['article_cat','Prediction'])
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+')
            self.log.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path, result.head().to_json(orient="records")

    def predictKeywords(self):
        try:
            self.log.log(self.file_object,'Start of Prediction')
            user_data=extract_article_text(self.pred_data_val,'') 
            user_data.deleteExistingOutputFiles()
            input=user_data.getTextFromFile()
            keywords_pred_val = keywords_prediction(input)
            keywords_res=keywords_pred_val.getKeywordsYake()
            #keywords_res=keywords_pred_val.getKeyWordsBert()
            result = pandas.DataFrame(keywords_res)
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",mode='a+')
            self.log.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path, keywords_res
   
    def validateDocumentFromModel(self):
        try:
            self.pred_data_val.deletePredictionFile()
            self.log.log(self.file_object,'Start of Prediction')
            data_getter=Data_Getter(self.file_object)
            data=data_getter.get_data()
            print(type(data))
            preprocessor=Preprocesser()
            # is_zerona_present,cols_with_nazerovalues=preprocessor.getColumns_with_NAZero(data)
            # if(is_zerona_present):
            #     data=preprocessor.remove_Columns(data,cols_with_nazerovalues)
            # data = preprocessor.replaceInvalidValuesWithNull(data)
            # is_null_present,cols_with_missing_values=preprocessor.isNull_Present(data)
            # if(is_null_present):
            #     data=preprocessor.impute_missing_values(data,cols_with_missing_values)
            X,Y=preprocessor.separate_label_feature(data,label_column_name='ArticleCategory')
            file_loader=File_Operation(self.file_object)
            Logistic=file_loader.load_model('Logistic')
            f_names=Logistic.feature_names
            self.log.log(self.file_object,f'The column values for the prediction data for logistic  is {f_names}')
            model_finder=Model_Finder(self.file_object)
            vec,traindata=model_finder.vectoring(X)
            new_data_tfidf =vec.transform(X)
            desired_shape = (1, 5000)
            sparse_matrix_5000=hstack([new_data_tfidf, csr_matrix((1, desired_shape[1] - new_data_tfidf.shape[1]))])
            print(sparse_matrix_5000.shape)
            # new_shape = (1, 5000)
            # reshaped_matrix = coo_matrix(new_data_tfidf).reshape(new_shape).tocsr()
            # print("Number of features in X_new_tfidf:", reshaped_matrix.shape[1])
            result=list(Logistic.predict(sparse_matrix_5000))
            print("Number of features expected by the model:", Logistic.coef_.shape[1])
            print("Vectorizer configuration:", vec.get_params())
            self.log.log(self.file_object,'End of Prediction')
            self.log.log(self.file_object,str(result[0])) 
            return result[0]
        except Exception as ex:
            self.log.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        


        


        
