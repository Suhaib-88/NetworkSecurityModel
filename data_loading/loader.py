import pandas as pd

class DataLoader:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object

    def get_training_data(self,path,col):
        try:
            train_df=pd.read_csv(path,index_col=col)
            self.logger_object.log(self.file_object,'Reading Training file')
        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while reading training file {e}")
        
        return train_df

    def get_testing_data(self,path,col):
        try:
            test_df=pd.read_csv(path,index_col=col)
            self.logger_object.log(self.file_object,'Reading Testing file')
        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while reading test file {e}")
        
        return test_df