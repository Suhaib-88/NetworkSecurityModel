# Importing modules
from Application_Logger.logger import App_Logger
from data_preprocessing.preprocessor import Preprocessor
from data_loading.loader import DataLoader
from model_building.models import ModelBuilding
from data_visualization.plotting import DataPlotting
from data_clustering.clustering import KMeansClustering

import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class MultiModel:
    def __init__(self):
        self.logger_obj=App_Logger()
        self.file_object= open("training_logs/multi_logs.txt","a+")
        

    def training(self):
        ## multi-class Classification
        loader= DataLoader(file_object=self.file_object,logger_object=self.logger_obj)
        train_data=loader.get_training_data(path='multiclass-DATA\kdd_train_final.csv',col=None)
        test_data=loader.get_testing_data(path="multiclass-DATA\kdd_test_final.csv",col=None)

        preprocess=Preprocessor(file_object= self.file_object,logger_object= self.logger_obj)
        
        train_data=preprocess.categorical_encoder(train_data)
        test_data=preprocess.categorical_encoder(test_data)

        corr_cols_train=preprocess.correlation(train_data,0.95)
        
        train_data= preprocess.column_dropper(train_data,corr_cols=corr_cols_train)
        test_data= preprocess.column_dropper(test_data,corr_cols= corr_cols_train)

        train_data=preprocess.drop_nans(train_data)
        test_data=preprocess.drop_nans(test_data)

        train_data=preprocess.label_encoder(train_data,label_name='attack_type')
        test_data=preprocess.label_encoder(test_data,label_name="attack_type")

        X_train,y_train= preprocess.sep_target_column(train_data,'attack_type')
        X_test,y_test=preprocess.sep_target_column(test_data,'attack_type')


        """ Applying the clustering approach"""
        kmeans=KMeansClustering(self.file_object,self.logger_obj) # object initialization.
        number_of_clusters=kmeans.elbow_plot(X_train)  #  using the elbow plot to find the number of optimum clusters
        # Divide the data into clusters
        X=kmeans.create_clusters(X_train,number_of_clusters)

        #create a new column in the dataset consisting of the corresponding cluster assignments.
        X['Labels']=y_train

        # getting the unique clusters from our dataset
        list_of_clusters=X['Cluster'].unique()

        """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""
        x_trains=[]
        y_trains=[]
        for i in list_of_clusters:
            cluster_data=X[X['Cluster']==i] # filter the data for one cluster
            # Prepare the feature and Label columns
            cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1).values
            cluster_label= cluster_data['Labels'].values
            
            x_trains.extend(cluster_features)
            y_trains.extend(cluster_label)

        number_of_clusters=kmeans.elbow_plot(X_test)  #  using the elbow plot to find the number of optimum clusters
        # Divide the data into clusters
        X=kmeans.create_clusters(X_test,number_of_clusters)

        #create a new column in the dataset consisting of the corresponding cluster assignments.
        X['Labels']=y_test

        # getting the unique clusters from our dataset
        list_of_clusters=X['Cluster'].unique()

        """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""
        x_tests=[]
        y_tests=[]
        for i in list_of_clusters:
            cluster_data=X[X['Cluster']==i] # filter the data for one cluster
            # Prepare the feature and Label columns
            cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1).values
            cluster_label= cluster_data['Labels'].values
            
            x_tests.extend(cluster_features)
            y_tests.extend(cluster_label)

        X_train,X_test=preprocess.scale_data(x_trains,x_tests)

        trainer=ModelBuilding(file_object= self.file_object,logger_object= self.logger_obj)
        
        classifier_name,training_time,testing_time,training_score,testing_score,y_pred=trainer.model_fitting([OneVsRestClassifier(LinearSVC(random_state=0)),OneVsRestClassifier(LogisticRegression()),MultinomialNB(),DecisionTreeClassifier(),AdaBoostClassifier()],X_train,y_trains,X_test,y_tests)
        
        plotter= DataPlotting(file_object= self.file_object,logger_object= self.logger_obj)
        plotter.plot(classifier_name,training_time,testing_time,training_score,testing_score,name='M')
        df=pd.DataFrame({"classifier_name":classifier_name,"training_time":training_time,"testing_time":testing_time,'training_score':training_score,'testing_score':testing_score})
        y_pred=y_pred.tolist()
        return df,y_pred,y_tests
        