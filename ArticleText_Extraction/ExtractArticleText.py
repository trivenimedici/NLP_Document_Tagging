from app_logger.logger import APP_LOGGER
import os
import csv
from docx import Document
import PyPDF2
import random
import numpy as np 




class extract_article_text:
    def __init__(self,fileName,input):
        self.fileObject=open("Log_Files_Collection/Prediction_Logs/Result_Log.txt","a+")
        self.log=APP_LOGGER()
        self.filename=fileName
        self.inputtext=input


    def generate_random_number(start, end):
        return random.randint(start, end)


    def get_document_type(self,filepath):
        try:
            self.log.log(self.fileObject,'Getting the document type')
            split_tup = os.path.splitext(filepath)
            file_name = split_tup[0]
            file_extension = split_tup[1]
            self.log.log(self.fileObject,f'the document type is {file_extension}')
            return file_extension
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e

    def convert_docx_to_text(self,filepath):
        try:
            self.log.log(self.fileObject,'Converting the docx file to text file')
            doc = Document(filepath)
            text_content = []
            for paragraph in doc.paragraphs:
                print(paragraph.text)
                if(paragraph.text not  in ''):
                    text_content.append(str(paragraph.text))
            print(text_content)
            # Save the extracted text into a new text file
            num=random.randint(1, 1000)
            global  output_file_name
            output_file_name='Prediction_Raw_files_validated\outputfiles\output'+str(num)+'.txt'
            with open(output_file_name, 'w', encoding='utf-8') as output_file:
                output_file.write('\n'.join(text_content))
            self.log.log(self.fileObject,'Converted the docx file to text file')
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e
    
    def convert_pdf_to_text(self,filepath):
        try:
            self.log.log(self.fileObject,'Converting the pdf file to text file')
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfFileReader(file)
                text = ''
                for page_num in range(pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)
                    text += page.extractText()
             # Save the extracted text into a new text file
            num=random.randint(1, 1000)
            global  output_file_name
            output_file_name='Prediction_Raw_files_validated\outputfiles\output'+str(num)+'.txt'
            with open(output_file_name, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            self.log.log(self.fileObject,'Converted the pdf file to text file')
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e
        
    def convert_csv_to_text(self,filepath):
        try:
            self.log.log(self.fileObject,'Converting the csv file to text file')
            with open(filepath, 'r') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
         # Save the CSV rows as text into a new text file
            num=random.randint(1, 1000)
            global  output_file_name
            output_file_name='Prediction_Raw_files_validated\outputfiles\output'+str(num)+'.txt'
            with open(output_file_name, 'w', encoding='utf-8') as output_file:
                for row in rows:
                    output_file.write(','.join(row) + '\n')
            self.log.log(self.fileObject,'Converted the csv file to text file')
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e

    def convert_file_to_text(self,filepath):
        try:
            self.log.log(self.fileObject,'Converting the file to text file')
            num=random.randint(1, 1000)
            global  output_file_name
            output_file_name='Prediction_Raw_files_validated\outputfiles\output'+str(num)+'.txt'
            with open(filepath) as my_file:
                text_content = my_file.readlines()
            with open(output_file_name, 'w', encoding='utf-8') as output_file:
                output_file.write('\n'.join(text_content))
            self.log.log(self.fileObject,'Converted the file to text file')
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e

    def getTextFromFile(self):
        try:
            self.log.log(self.fileObject,'Getting text from file')
            ARTICLE_TEXT=[]
            File_Paths=self.getFilePath()
            for file_path in File_Paths :
                file_ty = self.get_document_type(file_path)
                if(file_ty=='.csv'):
                    self.convert_csv_to_text(file_path)
                elif(file_ty=='.docx'):
                    self.convert_docx_to_text(file_path)
                elif(file_ty=='.pdf'):
                    self.convert_pdf_to_text(file_path)
                elif(file_ty=='.txt'):
                    self.convert_file_to_text(file_path)
                with open(output_file_name,'r') as f:
                        ARTICLE_TEXT.append(f.read().splitlines() )
            print(str(ARTICLE_TEXT))
            self.log.log(self.fileObject,f'Got the text from output files {ARTICLE_TEXT}')
            return ARTICLE_TEXT
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e
    
    def getTextFromFileByDir(self,file_path):
        try:
            self.log.log(self.fileObject,'Getting text from file')
            ARTICLE_TEXT=[]
            file_ty = self.get_document_type(file_path)
            if(file_ty=='.csv'):
                self.convert_csv_to_text(file_path)
            elif(file_ty=='.docx'):
                self.convert_docx_to_text(file_path)
            elif(file_ty=='.pdf'):
                self.convert_pdf_to_text(file_path)
            elif(file_ty=='.txt'):
                self.convert_file_to_text(file_path)
            with open(output_file_name,'r') as f:
                    ARTICLE_TEXT.append(f.read().splitlines() )
            print(str(ARTICLE_TEXT))
            self.log.log(self.fileObject,f'Got the text from output files {ARTICLE_TEXT}')
            return ARTICLE_TEXT
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e

    def createResultDir(self):
        try:
            path=os.path.join('Result_Dataset_File/','input_data/')
            if not os.path.isdir(path):
                os.makedirs(path)
        except OSError as e:
            file=open('ErrorLogs.txt','a+')
            self.logger.log(file,"Error while creating directory %s:"% e)
            file.close()
            raise OSError
        
    def getFilePath(self):
        try:
            File_Paths=[]
            self.log.log(self.fileObject,'Getting file path from batch files')
            root_dir='Batch_Prediction_Dataset'
            print(root_dir)
            for subdir, dirs, files in os.walk(root_dir):
                for file in files:
                    filepath =os.path.join(subdir, file)
                    print(filepath)
                    File_Paths.append(filepath)
                print(File_Paths)
            return File_Paths
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e

    def saveInputasOutputFile(self):
        try:
            self.log.log(self.fileObject,'Inserting input to file')
            num=random.randint(1, 1000)
            global  output_file_name
            output_file_name='Prediction_Raw_files_validated\outputfiles\output'+str(num)+'.txt'
            with open(output_file_name, 'w', encoding='utf-8') as output_file:
                output_file.writelines(self.inputtext)
            self.log.log(self.fileObject,'Converted the file to text file')
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e


    def extractUserDataFromArticle(self):
        try:
            ARTICLE_ID=[]
            ARTICLE_TEXT=[]
            ARTICLE_CATEGORY=[]
            KEYWORDS=[]
            NO_OF_KEYWORDS=[]
            ARTICLE_CAT=[]
            self.log.log(self.fileObject,'Extracting Text from Article')
            
            #return articletext
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e


    def deleteExistingInputFiles(self):
        dir_path='Batch_Prediction_Dataset/'
            # for f in os.listdir(dir_path):
            #     file=os.path.json(dir_path,f)
            #     if os.path.isfile(file):
            #         print('Deleting file: ',file)
            #         os.remove(file)
        try:
            files = os.listdir(dir_path)
            for filename in files:
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    self.log.log(self.fileObject,'Deleted: {file_path}')
        except FileNotFoundError:
            self.log.log(self.fileObject,"File not found: {file_path}")
        except Exception as e:
            self.log.log(self.fileObject,"Error deleting {file_path}: {e}")
            raise OSError
        
    def deleteExistingOutputFiles(self):
        dir_path='Prediction_Raw_files_validated/outputfiles/'
            # for f in os.listdir(dir_path):
            #     file=os.path.json(dir_path,f)
            #     if os.path.isfile(file):
            #         print('Deleting file: ',file)
            #         os.remove(file)
        try:
            files = os.listdir(dir_path)
            for filename in files:
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    self.log.log(self.fileObject,'Deleted: {file_path}')
        except FileNotFoundError:
            self.log.log(self.fileObject,"File not found: {file_path}")
        except Exception as e:
            self.log.log(self.fileObject,"Error deleting {file_path}: {e}")
            raise OSError
        
    def deleteExistingFiles(self,dir_path):
            # for f in os.listdir(dir_path):
            #     file=os.path.json(dir_path,f)
            #     if os.path.isfile(file):
            #         print('Deleting file: ',file)
            #         os.remove(file)
        try:
            files = os.listdir(dir_path)
            for filename in files:
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    self.log.log(self.fileObject,'Deleted: {file_path}')
        except FileNotFoundError:
            self.log.log(self.fileObject,"File not found: {file_path}")
        except Exception as e:
            self.log.log(self.fileObject,"Error deleting {file_path}: {e}")
            raise OSError


    