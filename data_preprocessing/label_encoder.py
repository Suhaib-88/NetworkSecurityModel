from sklearn.preprocessing import LabelEncoder

class labelEncoder:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
    
    def multi_encoder(self,data,label_name):
        try:
            self.attacks_types = {
            'normal': 'normal',
            'back': 'dos',
            'buffer_overflow': 'u2r',
            'ftp_write': 'r2l',
            'guess_passwd': 'r2l',
            'imap': 'r2l',
            'ipsweep': 'probe',
            'land': 'dos',
            'loadmodule': 'u2r',
            'multihop': 'r2l',
            'neptune': 'dos',
            'nmap': 'probe',
            'perl': 'u2r',
            'phf': 'r2l',
            'pod': 'dos',
            'portsweep': 'probe',
            'rootkit': 'u2r',
            'satan': 'probe',
            'smurf': 'dos',
            'spy': 'r2l',
            'teardrop': 'dos',
            'warezclient': 'r2l',
            'warezmaster': 'r2l',
            }
            data[label_name]=data[label_name].map(self.attacks_types)
            
            enc=LabelEncoder()
            data[label_name]=enc.fit_transform(data[label_name])
            
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,f"Error occured while encoding label :{e}")