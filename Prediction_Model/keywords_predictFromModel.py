from app_logger.logger import APP_LOGGER
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import yake

class keywords_prediction:
    def __init__(self,inputText):
        self.input=inputText
        self.fileObject=open("Log_Files_Collection/Prediction_Logs/Prediction_Main_Log.txt","a+")
        self.log=APP_LOGGER()

    def remove_duplicates_from_list(self,input_list):
        try:
            result_list = []
            for item in input_list:
                words = item.split()
                if len(words) == len(set(words)):
                # No duplicates in the words, add to the result list
                    result_list.append(item)
            return result_list
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e


    def getKeyWordsBert(self):
        try:
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(str(self.input))
            print(keywords)
            wrds=[x[0] for x in keywords]
            res=self.remove_duplicates_from_list(wrds)
            # ARTICLE_KEYWORDS.append(wrds)
            # ARTICLE_KEYWORDS_COUNT.append(len(wrds))
            # res =str(ARTICLE_KEYWORDS)
            return res
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e
    
    def getKeywordsTFID(self):
        try:
            tfidf = TfidfVectorizer(max_features=300, ngram_range = (2,2))
            keywords=[]
            Y = tfidf.fit_transform(str(self.input))
            feature_names = Y.get_feature_names_out()
            print("TF-IDF Matrix:")
            print(Y.toarray())
            print("\nFeature Names:")
            print(feature_names)
            for f in feature_names:
                print(f)
                keywords.append(f)
            return keywords
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e
    
    def getKeywordsYake(self):
        try:
            kw_extractor = yake.KeywordExtractor()
            self.log.log(self.fileObject,f'Got the text from output files {self.input}')
            for art in self.input:
                print(art)
                self.log.log(self.fileObject,f'Got the text from output files {art}')
                keywords = kw_extractor.extract_keywords(str(art))
                print(keywords)
                wrds=[x[0] for x in keywords]
                # ARTICLE_KEYWORDS.append(wrds)
                # ARTICLE_KEYWORDS_COUNT.append(len(wrds))
                #res =str(ARTICLE_KEYWORDS)
                res=self.remove_duplicates_from_list(wrds)
            return res
        except Exception as e:
            self.log.log(self.fileObject,e)
            raise e

        


    
                                


