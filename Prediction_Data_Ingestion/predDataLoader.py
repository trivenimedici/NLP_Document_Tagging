from app_logger.logger import APP_LOGGER
import pandas as pd
import csv
import os
import glob

class Data_Getter:
    def __init__(self,file_Object):
        self.Prediction_file='Prediction_FileFromDB/InputFile.csv'
        self.file_object=file_Object
        self.log=APP_LOGGER()
        self.badFilePath = "Prediction_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Prediction_Raw_files_validated/Good_Raw"
        self.outputFilePath = "Prediction_Raw_files_validated/outputfiles"
    
    def get_data(self):
        """
            Method Name: get_data
            Description: This method is to get the data from source
            Output:  A pandas DataFrame   
            On Failure: Raise Exception
            Written By: triveni
            Version: 1.0
            Revisions: None
        """
        self.log.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        goodFilesPath=self.goodFilePath
        badFilesPath=self.badFilePath
        try:
            csv_files = [f for f in os.listdir(self.outputFilePath) if f.endswith('.csv')]
            self.data=pd.DataFrame()
            for file in csv_files:
                file_path = os.path.join(self.outputFilePath, file)
                df = pd.read_csv(file_path)
                self.data=  pd.concat([self.data, df], ignore_index=True)
            self.log.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.log.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.log.log(self.file_object,'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()