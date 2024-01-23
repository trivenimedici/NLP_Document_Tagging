import pandas as pd
import numpy as np
from app_logger.logger import APP_LOGGER
from sklearn.impute import KNNImputer
from scipy import stats
import matplotlib.pyplot as plt
import os
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from contractions import contractions_dict
import re
import nltk



class Preprocesser:
    def __init__(self):
        self.file_object = open("Log_Files_Collection/Training_Logs/PreprocessorLog.txt", 'a+')
        self.log=APP_LOGGER()
        
    
    def remove_Columns(self,data,columns):
        """
            Method Name: remove_Columns
            Description: This method is to remove the columns from the pandas dataframes
            Output:  A pandas DataFrame after removing the specified columns   
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object,'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1)
            self.log.log(self.file_object,'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.log.log(self.file_object,'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise e
    
    
    def separate_label_feature(self, data, label_column_name):
        """
            Method Name: separate_label_feature
            Description: This method is to separate the features and a Label Coulmns.
            Output:  Returns two separate Dataframes, one containing features and the other containing Labels
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data['ArticleText']
            self.Y=data[label_column_name]
            self.log.log(self.file_object,'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()
    def dropUnnecessaryColumns(self,data,columnNameList):
        """
            Method Name: dropUnnecessaryColumns
            Description: This method is to drop unnecessary column values
            Output:  returns the data frame with the removed columns
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        data = data.drop(columnNameList,axis=1)
        return data

    def getTrainDataColumns(self):
        """
            Method Name: deleteColumnsBasedonTrain
            Description: This method is to get the column names from train file and remove the columns based on this
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            self.log.log(self.file_object,'Entered the getTrainDataColumns method of the Preprocessor class')
            for file in os.listdir('Prediction_Raw_files_validated/Good_Raw/'):
                csv=pd.read_csv('Prediction_Raw_files_validated/Good_Raw/'+file)
                train_columns=csv.columns
            return train_columns       
        except Exception as e:
            self.log.log(self.file_object, "Error Occured:: %s" % e)
            raise e

    def get_Uncommon_Columns(self,list1,list2):
        """
            Method Name: get_Uncommon_Columns
            Description: This method is to get the column names from train file and predit file which are not common
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            self.log.log(self.file_object,'Entered the get_Uncommon_Columns method of the Preprocessor class')
            col_to_remove =list(set(list1)-set(list2))
            return col_to_remove        
        except Exception as e:
            self.log.log(self.file_object, "Error Occured:: %s" % e)
            raise e


    def replaceInvalidValuesWithNull(self,data):
        """
            Method Name: replaceInvalidValuesWithNull
            Description: This method is to replace invalid values with null
            Output:  returns the data frame with the removed invalid values
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def isNull_Present(self,data):
        """
            Method Name: isNull_Present
            Description: This method is check if the null values are present
            Output:  Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values=[]
        self.cols = data.columns
        try:
            self.null_counts=data.isna().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present): 
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') 
            self.log.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def getColumns_with_NAZero(self,data):
        """
            Method Name: getColumns_with_NAZero
            Description: This method is check if the null values are present or zero values are present for the column and get all those column names
            Output:  Returns list of column names which has na or zero values
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the getColumns_with_NAZero method of the Preprocessor class')
        self.allzero_colms =[]
        self.null_present=False
        try:
            for column in data:
                if data[column].isna().all():
                    self.allzero_colms.append(column)
                    self.null_present=True
                        #print(allzero_colms)
            if(self.null_present): 
                self.dataframe_with_nazero = pd.DataFrame()
                self.dataframe_with_nazero['columns'] = data.columns
                self.dataframe_with_nazero['zero values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_nazero.to_csv('preprocessing_data/nazero_values.csv') 
            self.log.log(self.file_object,'Finding zero or na values is a success.Data written to the null values file. Exited the getColumns_with_NAZero method of the Preprocessor class')
            return self.null_present, self.allzero_colms
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in getColumns_with_NAZero method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object,'Finding missing values failed. Exited the getColumns_with_NAZero method of the Preprocessor class')
            raise Exception()

    def dropColumns_with_Constant(self,data):
        """
            Method Name: dropColumns_with_Constant
            Description: This method is check if the constant values are present for the column and drop all those column names
            Output:  data without constant values
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the dropColumns_with_Constant method of the Preprocessor class')
        try:
            data=data.drop(data.columns[data.nunique()==1],axis=1)
            self.log.log(self.file_object,'Finding constant values is a success.Data written to the constant values file. Exited the dropColumns_with_Constant method of the Preprocessor class')
            return data
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in dropColumns_with_Constant method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object,'Finding constant values failed. Exited the dropColumns_with_Constant method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self,data):
        """
            Method Name: encodeCategoricalValues
            Description: This method encodes all the categorical values in the data set
            Output:  A Dataframe which has all the categorical values encoded
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the encodeCategoricalValues method of the Preprocessor class')
        try:
            data["class"] = data["class"].map({'p': 1, 'e': 2})
            for column in data.drop(['class'],axis=1).columns:
                data = pd.get_dummies(data, columns=[column])
            self.log.log(self.file_object,'encoding the categorical values is a success.Data written to the constant values file. Exited the encodeCategoricalValues method of the Preprocessor class')
            return data
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in encodeCategoricalValues method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object,'Finding categorical columns failed. Exited the encodeCategoricalValues method of the Preprocessor class')
            raise Exception()

    def drop_Features_with_Coorelation(self,data):
        """
            Method Name: drop_Features_with_Coorelation
            Description: This method drops columns which has coleration with features
            Output:  A Dataframe which has all the correlation features dropped
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the drop_Features_with_Coorelation method of the Preprocessor class')
        try:
            cor_matrix = data.corr().abs()
            upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            data = data.drop(to_drop, axis=1)
            return data
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in drop_Features_with_Coorelation method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object,'Finding correlation features failed. Exited the drop_Features_with_Coorelation method of the Preprocessor class')
            raise Exception()

    def remove_outliers(self,data):
        """
            Method Name: remove_outliers
            Description: This method is to remove the outliers
            Output:  A Dataframe without outliers
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the remove_outliers method of the Preprocessor class')
        try:
            z=np.abs(stats.zscore(data))
            z_p=data[(z < 3).all(axis=1)]
            Q1=data.quantile(0.25)
            Q3=data.quantile(0.75)
            IQR=Q3-Q1
            lowqe_bound=Q1 - 1.5 * IQR
            upper_bound=Q3 + 1.5 * IQR
            IQR_p = data[~((data < lowqe_bound) |(data > upper_bound)).any(axis=1)]
            data=pd.DataFrame(IQR_p)
            return data
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in remove_outliers method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object,'Finding outliers failed. Exited the remove_outliers method of the Preprocessor class')
            raise Exception()


    def impute_missing_values(self,data,cols_with_missing_values):
        """
            Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using KNN Imputer
            Output:  A Dataframe which has all the missing values imputed
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        self.cols_with_missing_values=cols_with_missing_values
        try:
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
                self.log.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            imputer=KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data)
            self.new_data=pd.DataFrame(data=self.new_array,columns=self.data.columns)
            self.log.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):
        """
            Method Name: get_columns_with_zero_std_deviation
            Description: This method finds out the columns which have a standard deviation of zero
            Output:  List of the columns with standard deviation of zero
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                 if (self.data_n[x]['std'] == 0):
                    self.col_to_drop.append(x) 
            self.log.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.log.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()
        
    def converttoLowercase(self,input):
        """
            Method Name: converttoLowercase
            Description: This method Convert all text to lowercase to ensure consistency. This helps in treating words with different cases as the same.
            Output:  text after converting to lower cases
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            text=input.lower()
            self.log.log(self.file_object, 'converted to lower case')
            return text
        except Exception as e:
            self.log.log(self.file_object, 'exception while converting to lower case')
            raise Exception()

    

    def Tokenization(self,input):
        """
            Method Name: Tokenization
            Description: This method Break text into individual words or tokens. Tokenization is essential for many NLP tasks as it helps in analyzing the structure of sentences.
            Output:  input text after tokenization
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            text=word_tokenize(input)
            self.log.log(self.file_object, 'tokenized the inputtext')
            return text
        except Exception as e:
            self.log.log(self.file_object, 'exception while tokenizing the inputtext {e}')
            raise Exception()
    

    def removePunctuationSpecialCharacters(self,input):
        """
            Method Name: removePunctuationSpecialCharacters
            Description: This method Eliminate punctuation marks and special characters, as they may not contribute much to the meaning of the text.
            Output:  input text after removing punctuations and special characters
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            text= re.sub(r'[^\w\s]', '', input)
            self.log.log(self.file_object, 'removed punctuations and special characters from the inputtext')
            return text
        except Exception as e:
            self.log.log(self.file_object, 'error while removing punctuations and special characters from the inputtext')
            raise Exception()
    

    def removeStopWords(self,input):
        """
            Method Name: removeStopWords
            Description: This method Remove common words like "the," "is," and "and" (stopwords) as they often do not carry significant meaning.
            Output:  input text after removing stopwords
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in input if word not in stop_words]
            self.log.log(self.file_object, 'removed stop words from the inputtext')
            return tokens
        except Exception as e:
            self.log.log(self.file_object, 'error while removing stopwords from the inputtext')
            raise Exception()
    

    def stemming(self,input):
        """
            Method Name: stemming
            Description: This method Reduce words to their root form to normalize the text.
            Output:  inputtext after stemming
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            stemmer = PorterStemmer()
            stemmed_tokens = [stemmer.stem(word) for word in input]
            self.log.log(self.file_object, 'stemming the inputtext')
            return stemmed_tokens
        except Exception as e:
            self.log.log(self.file_object, 'error while stemming the inputtext')
            raise Exception()


    def lemmatization(self,input):
        """
            Method Name: lemmatization
            Description: This method lemmatization help in reducing inflected words to a common base form.
            Output:  input text after lemmatization
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in input]
            self.log.log(self.file_object, 'lemmatizing the inputtext')
            return lemmatized_tokens
        except Exception as e:
            self.log.log(self.file_object, 'error while lemmatizing the inputtext')
            raise Exception()
    
    def handlingNumbers(self,input):
        """
            Method Name: handlingNumbers
            Description: This method replace numbers with a placeholder token or remove them altogether.
            Output:  input text after removing numbers
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            text = re.sub(r'\d+', '', input) 
            self.log.log(self.file_object, 'removing numbers from the inputtext') 
            return text
        except Exception as e:
            self.log.log(self.file_object, 'error while removing numbers from the inputtext')
            raise Exception()
    
    def handlingHTMLTagsNUrls(self,input):
        """
            Method Name: handlingHTMLTagsNUrls
            Description: This method removes html tags and urls
            Output:  input text after removing html tags and urls
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            text = re.sub(r'<.*?>', '', input)  # Remove HTML tags
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            self.log.log(self.file_object, 'removed html tags and urls from the inputtext')
            return text
        except Exception as e:
            self.log.log(self.file_object, 'error while removing html tags and urls from the inputtext')
            raise Exception()
    
    def handlingContractions(self,input):
        """
            Method Name: handlingContractions
            Description: This method is for Expand contractions to ensure consistency in the text. For example, replace "can't" with "cannot."
            Output:  input text after handling contractions
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            for contraction, expansion in contractions_dict.items():
                text = re.sub(contraction, expansion, input, flags=re.IGNORECASE)
            self.log.log(self.file_object, 'handling contractions from the inputtext')
            return text
        except Exception as e:
            self.log.log(self.file_object, 'error while handling contractions from the inputtext')
            raise Exception()
    
    def handlingEmoticonsSpecialCharacters(self,input):
        """
            Method Name: handlingEmoticonsSpecialCharacters
            Description: This method Depending on your task, you may decide to remove or replace emoticons and special characters.
            Output:  input text after removing emoticons and special characters
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            text = re.sub(r':\)|:-\)|;\)|:-\(', 'smile', input)
            self.log.log(self.file_object, 'removing emoticons and special characters from the inputtext')
            return text
        except Exception as e:
            self.log.log(self.file_object, 'error while removing emoticons and special characters from the inputtext')
            raise Exception()
    
    def handlingWhiteSpaces(self,input):
        """
            Method Name: handlingWhiteSpaces
            Description: This method Remove extra white spaces and normalize spaces between words.
            Output:  input text after removing white spaces
            On Failure: Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            text = re.sub(' +', ' ', input)
            text = text.strip() 
            self.log.log(self.file_object, 'removing white spaces from the inputtext')
            return text
        except Exception as e:
            self.log.log(self.file_object, 'error while removing white spaces from the inputtext')
            raise Exception()


    def dataPreprocessing(self,input):
        try:
            flat_data = ' '.join([item for sublist in input for item in sublist])
            data=self.converttoLowercase(flat_data)
           # data = self.Tokenization(data)
            data=self.removePunctuationSpecialCharacters(data)
           # data=self.removeStopWords(data)
          #  data=self.stemming(data)
          #  data=self.lemmatization(data)
          #  data=self.handlingNumbers(data)
           # data=self.handlingHTMLTagsNUrls(data)
          #  data=self.handlingContractions(data)
          #  data=self.handlingEmoticonsSpecialCharacters(data)
          #  data=self.handlingWhiteSpaces(data)
            self.log.log(self.file_object, 'preprocessing completed')
            return data
        except Exception as e:
            self.log.log(self.file_object, 'error while preprocessing the inputtext {e}')
            raise Exception()



