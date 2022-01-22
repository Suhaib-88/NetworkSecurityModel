import time

classifier_name=[]
training_time=[]
testing_time=[]
training_score=[]
testing_score=[]

class ModelBuilding:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
    

    def model_fitting(self,classifier:list,X_train,y_train,X_test,y_test): 
        f1=[]
        pr=[]
        rec=[]
        try:            
            self.logger_object.log(self.file_object,f"Model Training: Starting")
            for i in range(len(classifier)):
                start_time = time.time()
                classifier[i].fit(X_train,y_train)
                end_time = time.time()
                training_time.append(end_time-start_time)
                print(f"training_time added")

                start_time = time.time()
                yPred=classifier[i].predict(X_test)
                end_time = time.time()
                testing_time.append(end_time-start_time)
                print(f"testing_time added")

                training_score.append(classifier[i].score(X_train,y_train))
                print(f"training_score added")


                testing_score.append(classifier[i].score(X_test,y_test))
                print(f"testing_score added")
                classifier_name.append(str(classifier[i])[:22])
                
            self.logger_object.log(self.file_object,f"Model Training: Completed")
       
        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while training model:{e}")
        return classifier_name,training_time,testing_time,training_score,testing_score,yPred       
