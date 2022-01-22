from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self,file_object,logger_object):
        self.file_object= file_object
        self.logger_object=logger_object

    def removing_quotes(self,df):
        try:
            a=[]
            for i in df.columns:
                a.append(i.split("'")[1])
            df.columns=a
            self.logger_object.log(self.file_object,f"Sucessfully removed quotes")
        
        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while removing quotes {e}")
        
        return df
        

    def categorical_encoder(self,df):
        try:
            pmap = {'icmp':0, 'tcp':1, 'udp':2}
            df['protocol_type'] = df['protocol_type'].map(pmap)

            fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}
            df['flag'] = df['flag'].map(fmap)
            self.logger_object.log(self.file_object,f"Sucessfully encoded categorical columns")

        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while encoding categories {e}")
        
        return df


    def correlation(self,df, threshold):
        try:
            col_corr = set()  # Set of all the names of correlated columns
            corr_matrix = df.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                        colname = corr_matrix.columns[i]  # getting the name of column
                        col_corr.add(colname)
            
            self.logger_object.log(self.file_object,f"Performed correlation")
        
        except Exception as e:
            self.logger_object.log(self.file_object,f"Unable to perform correlation: {e}")
        return list(col_corr)

    
    def column_dropper(self,df,corr_cols):
        try:    
            df.drop('service',axis=1,inplace=True)

            df.drop(corr_cols,axis=1,inplace=True)

            df=df[[i for i in df.columns if df[i].nunique() > 1]]
            
            self.logger_object.log(self.file_object,f"dropping uncessary columns")

        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while dropping columns {e}")
        
        return df


    def drop_nans(self,df):
        try:
            df.dropna(axis=0,inplace=True)
            df=df[df!='']
            self.logger_object.log(self.file_object,f"dropping empty or nan values")

        except Exception as e:
            self.logger_object.log(self.file_object,f"Error dropping na values :{e}")
        
        return df


    def label_encoder(self,df,label_name):
        try:
            self.encoder=LabelEncoder()
            df[label_name]=self.encoder.fit_transform(df[label_name])
            self.logger_object.log(self.file_object,f"label encoding target")

        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while encoding label :{e}")
        
        return df


    def sep_target_column(self,df,target_col):
        try:
            self.data=df
            self.target_col= target_col
            self.X= self.data.drop(columns=[self.target_col],axis=1)
            self.y=  self.data[self.target_col]
            self.logger_object.log(self.file_object,f"Seperating target col")

        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while seperating target feat: {e}")
        
        return self.X,self.y


    def scale_data(self,X_train,X_test):
        try:
            self.scaler=MinMaxScaler()
            self.scaler.fit(X_train)
            X_train=self.scaler.transform(X_train)
            X_test=self.scaler.transform(X_test)
            self.logger_object.log(self.file_object,f"scaling train and test data")

        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while scaling data:{e}")
        
        return X_train,X_test
    

