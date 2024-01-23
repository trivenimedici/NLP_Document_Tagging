from app_logger.logger import APP_LOGGER
import os
import pandas as pd
import json
import shutil
import re
import datetime
import csv
from werkzeug.utils import secure_filename
from pathlib import Path
from ArticleText_Extraction import ExtractArticleText  

class RawValidations:
    def __init__(self,path):
        self.filepath=path
        self.schema_path="prediction_schema.json"
        self.logger=APP_LOGGER()
    

    def valuesFromDataSchema(self):
        """
            Method Name: valuesFromDataSchema
            Description: This method extracts all the relevant information from the pre-defined "Schema" file.
            Output:  column names, Number of Columns
            On Failure: Raise ValueError,KeyError,Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            with open(self.schema_path,'r') as f:
                dic=json.load(f)
                f.close()
            no_of_columns = dic['NoOfColumns']
            col_names=dic['columnnames']
            file=open('Log_Files_Collection/Prediction_Logs/schemaValidationsLogs.txt','a+')
            self.logger.log(file,f"the number of coumns are {no_of_columns} and the column names are {col_names}")
            file.close()
        except ValueError:
            file=open('Log_Files_Collection/Prediction_Logs/schemaValidationsLogs.txt','a+')
            self.logger.log(file,"ValueError:Value not found inside the schema.json")
            raise ValueError
        except KeyError:
            file=open('Log_Files_Collection/Prediction_Logs/schemaValidationsLogs.txt','+a')
            self.logger.log(file,"KeyValueError:Key value error incorrect key passed")
            raise KeyError
        except Exception as e:
            file=open('Log_Files_Collection/Prediction_Logs/schemaValidationsLogs.txt','+a')
            self.logger.log(file,str(e))
            raise e
        return no_of_columns,col_names
                
    def regexFileCreation(self):
        """
            Method Name: regexFileCreation
            Description: This method is to create regex based on the file name given in schema file
                Output:  manual_regex_pattern
            On Failure: None
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        manual_regex_pattern="dataset_[a-z0-9*]+.csv"
        return manual_regex_pattern
    
    def validationFileNameRaw(self,regex):
        """
            Method Name: validationFileNameRaw
            Description: This method is to validate the file name for the given input datasets
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.deleteExistingGoodDataPredictionFolder()
        self.deleteExistingBadPredictionDataFiles()
       # self.deleteExistingResultPredictionDataFiles()
        self.createDirForGoodBadData()
        onlyfiles=[f for f in os.listdir(self.filepath)]
        try:
            file=open('Log_Files_Collection/Prediction_Logs/nameValidationLog.txt','a+')
            for filename in onlyfiles:
                if(re.match(regex,filename)):
                    shutil.copy(self.filepath+"/" + filename, "Prediction_Raw_files_validated/Good_Raw")
                    self.logger.log(file,f"Valid File name!! File moved to GoodRaw Folder :: {filename}")
                else:
                    shutil.copy("Batch_Prediction_DataSet/" + filename, "Prediction_Raw_files_validated/Bad_Raw")
                    self.logger.log(file,f"Invalid File Name!! File moved to Bad Raw Folder :: {filename}")
            file.close()
        except Exception as e:
            f = open("Log_Files_Collection/Prediction_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(f, f"Error occured while validating FileName {e}")
            f.close()
            raise e
    
    def validateColumnLength(self,NumberofColumns):
        """
            Method Name: validateColumnLength
            Description: This method is to validate number of columns for the input dataset file
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            f = open("Log_Files_Collection/Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f,"Column Length Validation started!!")
            for file in os.listdir('Prediction_Raw_files_validated/Good_Raw/'):
                csv=pd.read_csv('Prediction_Raw_files_validated/Good_Raw/'+file)
                if csv.shape[1]==NumberofColumns:
                    pass
                else:
                    shutil.move("Prediction_Raw_files_validated/Good_Raw/" + file, "Prediction_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.logger.log(f, "Column Length Validation Completed!!")
        except OSError:
            f = open("Log_Files_Collection/Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Log_Files_Collection/Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e
        f.close()
    def deletePredictionFile(self):
        if os.path.exists('Prediction_Output_File/Predictions.csv'):
            os.remove('Prediction_Output_File/Predictions.csv')


        


    def validatingMissingValuesinWholeColumn(self):
        """
            Method Name: validatingMissingValuesinWholeColumn
            Description: This method is to value the missing values in the column for the given input dataset file
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            f = open("Log_Files_Collection/Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f,"Missing Values Validation Started!!")
            for file in os.listdir('Prediction_Raw_files_validated/Good_Raw/'):
                csv=pd.read_csv('Prediction_Raw_files_validated/Good_Raw/'+file)
                count=0
                for columns in csv:
                    if((len(csv[columns])-csv[columns].count())==len(csv[columns])):
                        count+=1
                        shutil.move("Prediction_Raw_files_validated/Good_Raw/" + file,"Prediction_Raw_files_validated/Bad_Raw")
                        self.logger.log(f,"Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
                if count==0:
                   # csv.rename(columns={"Unnamed: 0": "phishing"}, inplace=True)
                    csv.to_csv("Prediction_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
        except OSError:
            f = open("Log_Files_Collection/Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Log_Files_Collection/Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e
        f.close()

    def createDirForGoodBadOutputData(self):
        """
            Method Name: createDirForGoodBadData
            Description: This method creates directories to store the Good Data and Bad Data after validating the Prediction data.
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            path=os.path.join('Prediction_Raw_files_validated/','Good_Raw/')
            if not os.path.isdir(path):
                os.makedirs(path)
            path=os.path.join('Prediction_Raw_files_validated/','Bad_Raw/')
            if not os.path.isdir(path):
                os.makedirs(path)
            path=os.path.join('Prediction_Raw_files_validated/','input_data/')
            if not os.path.isdir(path):
                os.makedirs(path)
            path=os.path.join('Prediction_Raw_files_validated/','outputfiles/')
            if not os.path.isdir(path):
                os.makedirs(path)
        except OSError as e:
            file=open('ErrorLogs.txt','a+')
            self.logger.log(file,"Error while creating directory %s:"% e)
            file.close()
            raise OSError
    
    def deleteExistingGoodDataPredictionFolder(self):
        """
            Method Name: deleteExistingGoodDataPredictionFolder
            Description: This method deletes the directory made  to store the Good Data after loading the data in the table. Once the good files are loaded in the DB,deleting the directory ensures space optimization.
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            path='Prediction_Raw_files_validated/'
            if os.path.isdir(path+'Good_Raw/'):
                shutil.rmtree(path+'Good_Raw/')
                file=open('Log_Files_Collection/Prediction_Logs/GeneralLog.txt','a+')
                self.logger.log(file,"Good Raw directory deleted successfully!!")
                file.close()
        except Exception as e:
            file=open("Log_Files_Collection/Prediction_Logs/ErrorLogs.txt", 'a+')
            self.logger.log(file,"Error while deleting directory : %s" %e)
            file.close()
            raise e
    def deleteExistingOutputDataPredictioFolder(self):
        """
            Method Name: deleteExistingGoodDataTrainingFolder
            Description: This method deletes the directory made  to store the Good Data after loading the data in the table. Once the good files are loaded in the DB,deleting the directory ensures space optimization.
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            path='Prediction_Raw_files_validated/'
            if os.path.isdir(path+'outputfiles/'):
                shutil.rmtree(path+'outputfiles/')
                file=open('Log_Files_Collection/Prediction_Logs/GeneralLog.txt','a+')
                self.logger.log(file,"Good Raw directory deleted successfully!!")
                file.close()
        except Exception as e:
            file=open("Log_Files_Collection/Prediction_Logs/ErrorLogs.txt", 'a+')
            self.logger.log(file,"Error while deleting directory : %s" %e)
            file.close()
            raise e
        
    def deleteExistingBadPredictionDataFiles(self):
        """
            Method Name: deleteExistingBadPredictionDataFiles
            Description: This method deletes the directory made  to store the bad Data after loading the data in the table. Once the bad files are loaded in the DB,deleting the directory ensures space optimization.
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        try:
            path='Prediction_Raw_files_validated/'
            if os.path.isdir(path+'Good_Raw/'):
                shutil.rmtree(path+'Good_Raw/')
                file=open('Log_Files_Collection/Prediction_Logs/GeneralLog.txt','a+')
                self.logger.log(file,"Bad Raw directory deleted successfully!!")
                file.close()
        except OSError as e:
            file=open("Log_Files_Collection/Prediction_Logs/ErrorLogs.txt", 'a+')
            self.logger.log(file,"Error while deleting directory : %s" %e)
            file.close()
            raise e
    def moveBadFilesToArchiveBad(self):
        """
            Method Name: moveBadFilesToArchiveBad
            Description: This method is to move all the bad data files to archieve folder
            Output:  None
            On Failure: OsError
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.time = self.now.strftime("%H%M%S")
        try:

            source = 'Prediction_Raw_files_validated/Bad_Raw/'
            if os.path.isdir(source):
                path = "PredictionArchiveBadData"
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = 'PredictionArchiveBadData/BadData_' + str(self.date)+"_"+str(self.time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                file = open("Log_Files_Collection/Prediction_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file,"Bad files moved to archive")
                path = 'Prediction_Raw_files_validated/'
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + 'Bad_Raw/')
                self.logger.log(file,"Bad Raw Data Folder Deleted successfully!!")
                file.close()
        except Exception as e:
            file = open("Log_Files_Collection/Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)
            file.close()
            raise e

    def get_filepaths_from_gooddata(self):
        try:
            File_Paths=[]
            root_dir="Prediction_Raw_files_validated/Good_Raw/"
            for subdir, dirs, files in os.walk(root_dir):
                for file in files:
                        filepath =os.path.join(subdir, file) 
                        File_Paths.append(filepath)
            return File_Paths
        except Exception as e:
            file = open("Log_Files_Collection/Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while getting file paths:: %s" % e)
            file.close()
            raise e   

    def get_text_from_article(self):
        try:
            ARTICLE_ID=[]
            ARTICLE_CATEGORY=[]
            ARTICLE_TEXT=[]
            File_paths=self.get_filepaths_from_gooddata()
            for file_path in File_paths:
                ARTICLE_ID.append(file_path.split('/')[-1])
                ARTICLE_CATEGORY.append(file_path.split('/')[-2]) 
                with open(file_path, encoding='latin') as f:
                    ARTICLE_TEXT.append(f.read())
            self.logger.log(file, "Completed the article text extraction %s" % ARTICLE_TEXT)
            return ARTICLE_TEXT
        except Exception as e:
            file = open("Log_Files_Collection/Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while getting article text:: %s" % e)
            file.close()
            raise e

    def get_article_category(self):
        try:
            ARTICLE_CAT=[]
            ARTICLE_CATEGORY=[]
            self.get_text_from_article()
            for cat in ARTICLE_CATEGORY:
                txt = cat.replace('.','_')
                ARTICLE_CAT.append(txt) 
            self.logger.log(file, "Completed the article category extraction %s" % ARTICLE_CAT)
            return ARTICLE_CAT
        except Exception as e:
            file = open("Log_Files_Collection/Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while getting article category:: %s" % e)
            file.close()
            raise e
        
    def get_csv_outputfile(self):
        try:
            ARTICLE_TXT = self.get_text_from_article()
            ARTICLE_CATE = self.get_article_category()
            data = list(zip(ARTICLE_TXT, ARTICLE_CATE))
            csv_file = 'Training_Raw_Files_Validations/Prediction_Raw_files_validated/outputfiles/train_data.csv'
            with open(csv_file,'w',newline='') as file:
                writer=csv.writer(file)
                writer.writerows(data)
            self.logger.log(file,"created output file successfully")
        except Exception as e:
            file = open("Log_Files_Collection/Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while creating output file:: %s" % e)
            file.close()
            raise e

    def allowed_file(self):
        ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx', 'doc','dat','csv','octet-stream','','DAT','file'])
        return ALLOWED_EXTENSIONS
    
    
    def validate_input_files(self):
        self.deleteExistingGoodDataPredictionFolder()
        self.deleteExistingBadPredictionDataFiles()
        self.deleteExistingOutputDataPredictioFolder()
        #self.deleteExistingResultPredictionDataFiles()
        self.createDirForGoodBadOutputData()
        try:
            #root_dir='Batch_Training_Dataset'
           # files=os.listdir(self.filepath)
            for subdir, dirs, files in os.walk(self.filepath):
                ARTICLE_CATEGORY=[]
                ARTICLE_TEXT=[]
                for file in files:
                    file_path = os.path.join(subdir, file)
                    article_cat=file_path.split('\\')[-2]
                    article_cat=article_cat.split('.')[-1]
                    ARTICLE_CATEGORY.append(article_cat)
                    with open(file_path, encoding='latin') as f:
                        ARTICLE_TEXT.append(f.read())
                    ##extract_text=ExtractArticleText.extract_article_text(file_path,'')
                    #ARTICLE_TEXT=extract_text.getTextFromFile()
                    path_obj = Path(file_path)
                    file_extension = path_obj.suffix
                    file_extension = file_extension[1:]
                    if file and file_extension in self.allowed_file():
                        #filename = secure_filename(file)
                        shutil.copy(file_path, "Prediction_Raw_files_validated/Good_Raw")
                    else:
                        #filename = secure_filename(file)
                        shutil.copy(file_path, "Prediction_Raw_files_validated/Bad_Raw")
                if len(ARTICLE_TEXT)>0:
                    headers = ["ArticleText", "ArticleCategory"]
                    data = list(zip(ARTICLE_TEXT,ARTICLE_CATEGORY))
                    csv_file = 'Prediction_Raw_files_validated/outputfiles/'+ARTICLE_CATEGORY[0]+'_predict_data.csv'
                    with open(csv_file,'w',newline='') as file:
                        writer=csv.writer(file)
                        writer.writerow(headers)
                        writer.writerows(data)
        except Exception as e:
            f = open("Log_Files_Collection/Prediction_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(f, "Error occured while validating FileName %s" % e)
            f.close()
            raise e

         


