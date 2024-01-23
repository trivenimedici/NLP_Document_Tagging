from flask_cors import CORS, cross_origin
from flask import Flask,request, render_template,Response
import os
import json
import flask_monitoringdashboard as dashboard
import shutil
from ArticleText_Extraction.ExtractArticleText import  extract_article_text
from Prediction_Validations.predictionValidation import pred_validation
from Prediction_Model.predictFromModel import prediction
from Training_Validations.trainingValidation import train_validation
from Training_Model.trainingModel import Model_Training
from Prediction_Model.keywords_predictFromModel import keywords_prediction
from Data_Preprocessing.preprocessing import Preprocesser
from werkzeug.utils import secure_filename
import csv


os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx', 'doc','dat','csv','octet-stream','','DAT','file'])
UPLOAD_FOLDER='Batch_Prediction_Dataset'

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
dashboard.bind(app)
CORS(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/",methods=['POST','GET'])
@cross_origin()
def home():
    try:
        if request.method == 'POST':
            button_clicked = request.form['submit_button']
            inputtext = request.form['textcontent'] 
            #print(inputtext)
            if (inputtext != ''):
                print(inputtext) 
                user_data=extract_article_text('',inputtext)  
                user_data.deleteExistingInputFiles()
                user_data.deleteExistingOutputFiles()
                user_data.saveInputasOutputFile()   
                input=user_data.getTextFromFile() 
                print(input)  
            elif ('file' in request.files):
                print('in files logic')
                file=request.files['file']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    user_data=extract_article_text(filename,'') 
                    user_data.deleteExistingInputFiles()
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    print('file saved in '+UPLOAD_FOLDER+' as '+ filename)
                    user_data.deleteExistingOutputFiles()
                    input=user_data.getTextFromFileByDir(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 

                    print(input)  
                else:
                    return render_template('error.html') 
            user_data.createResultDir()
            if(input=='invalid'):
                return render_template('error.html')
            elif button_clicked == 'extractKeywords':
                keywords_pred_val = keywords_prediction(input)
                keywords_res=keywords_pred_val.getKeywordsYake()
                #keywords_res=keywords_pred_val.getKeyWordsBert()
                if os.path.isdir('Result_Dataset_File/input_data/'):
                    shutil.rmtree('Result_Dataset_File/input_data/')
                return render_template('keywordsresults.html',prediction=keywords_res)
            elif button_clicked == 'classify':
                #get input
                # preprocessor=Preprocesser()
                # input=preprocessor.dataPreprocessing(input)
                #create predict file
                user_data=extract_article_text('',input)  
                dir_path='predictDocument/'
                out_dir='Prediction_Raw_files_validated/outputfiles'
                user_data.deleteExistingFiles(dir_path)
                user_data.deleteExistingFiles(out_dir)
                headers = ["ArticleText", "ArticleCategory"]
                #data=list(zip([[input]],''))
                csv_file = 'Prediction_Raw_files_validated/outputfiles/predict_data.csv'
                with open(csv_file,'w',newline='') as file:
                    writer=csv.writer(file)
                    writer.writerow(headers)
                    writer.writerows([[input],''])
                # pred_val = pred_validation(dir_path)
                # pred_val.pred_validation()
                pred= prediction(out_dir)
                result = pred.validateDocumentFromModel()
                if os.path.isdir('Result_Dataset_File/input_data/'):
                    shutil.rmtree('Result_Dataset_File/input_data/')
                return render_template('classificationresults.html',classification=result)
        else:
            return render_template('index.html')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)
    


@app.route("/trainclassification", methods=['POST', 'GET'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json['folderPath'] is not None:
            path=request.json['folderPath']
            train_val = train_validation(path)
            train_val.train_validation()
            train_model= Model_Training()
            train_model.trainingModel()
        elif request.form is not None:
            path = request.form['filepath']
            train_val = train_validation(path)
            train_val.train_validation()
            train_model= Model_Training()
            train_model.trainingModel()
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" % ValueError) 
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


@app.route("/predictclassification",methods=['POST', 'GET'])
@cross_origin()
def classificationpredictRountClient():
    user_data=extract_article_text("Prediction_Output_File",'') 
   # user_data.deleteExistingFiles()
    try:
        if request.json is not None:
            filename=request.json['filepath']    
            pred= prediction(filename)
            #path,json_predictions = pred.predictKeywords()
            pred_val = pred_validation(filename)
            pred_val.pred_validation()
            # pred= prediction(path)
            path,json_predictions = pred.predictionFromModel()
            return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
        elif request.form is not None:
            path = request.form['filepath']
            pred_val = pred_validation(path)
            pred_val.pred_validation()
            pred= prediction(path)
            path,json_predictions = pred.predictionFromModel()
            return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)


@app.route("/predictkeywords",methods=['POST', 'GET'])
@cross_origin()
def predictRountClient():
    try:
        user_data=extract_article_text("Prediction_Output_File",'') 
        user_data.deleteExistingFiles()
        if request.json is not None:
            filename=request.json['filepath']    
            pred= prediction(filename)
            path,json_predictions = pred.predictKeywords()
            return Response("Prediction File created at "  +str(path) +' and the keywords for the document are '+str(json_predictions))
        elif request.form is not None:
            filename = request.form['filepath']
            pred= prediction(filename)
            path,json_predictions = pred.predictKeywords()
            return Response("Prediction File created at "  +str(path) +' and the keywords for the document are '+str(json_predictions))
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

# @app.route("/result",methods=['POST', 'GET'])
# @cross_origin()
# def showResult():
#     try:
#         if request.method == 'POST':
#             url = request.form['websiteurl']
#             user_data=extract_url_properties(url)
#             input_data_path=user_data.load_user_data()
#             if input_data_path == 'invalid url':
#                 return render_template('error.html')
#             else:
#                 pred_val = pred_validation(input_data_path)
#                 pred_val.pred_validation()
#                 pred= prediction(path)
#                 path,json_predictions,result = pred.predictionFromModel()
#                 if result == 0:
#                     res='It Looks like a legitimate  website!!'
#                 else:
#                     res='Its Looks like a Phishing Website!! Be cautious to access '
#                 return render_template('results.html')
#         else:
#             return render_template('index.html')
#     except ValueError:
#         return Response("Error Occurred! %s" %ValueError)
#     except KeyError:
#         return Response("Error Occurred! %s" %KeyError)
#     except Exception as e:
#         return Response("Error Occurred! %s" %e)
port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    # host = '0.0.0.0'
    # httpd = simple_server.make_server(host, port, app)
    # httpd.serve_forever()
    app.run(debug=True)
