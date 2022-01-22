import matplotlib.pyplot as plt
import seaborn as sns

class DataPlotting:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_obj=logger_object

    def plot(self,classifier_name,training_time,testing_time,training_score,testing_score,name):
        try:
            plt.figure(figsize=(4,4))
            ax=sns.barplot(classifier_name,training_time)
            ax.bar_label(ax.containers[0])
            plt.title('training time taken')
            plt.xticks(rotation=90)
            plt.savefig(f'static/trainingTime_{name}.jpg',bbox_inches='tight')        
            
            
            plt.figure(figsize=(4,4))
            ax=sns.barplot(classifier_name,testing_time)
            ax.bar_label(ax.containers[0])
            plt.title('testing time taken')
            plt.xticks(rotation=90)
            plt.savefig(f'static/testingTime_{name}.jpg',bbox_inches='tight')        
            
            plt.figure(figsize=(4,4))
            ax=sns.barplot(classifier_name,training_score)
            ax.bar_label(ax.containers[0])
            plt.title('training score')
            plt.xticks(rotation=90)
            plt.savefig(f'static/trainingScore_{name}.jpg',bbox_inches='tight')        
            
            plt.figure(figsize=(4,4))
            ax=sns.barplot(classifier_name,testing_score)
            ax.bar_label(ax.containers[0])
            plt.title('testing score')
            plt.xticks(rotation=90)
            plt.savefig(f'static/testingScore_{name}.jpg',bbox_inches='tight')

            
            self.logger_obj.log(self.file_object,f"Successfully saved plots")
        except Exception as e:
            self.logger_obj.log(self.file_object,f"Unable to plot graph :{e}")