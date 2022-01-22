import binaryClassification,multiClassification
from flask import Flask, request, render_template
import pandas as pd
from PredictionValidation.db_operation import DBoperator

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/binary_train", methods=['POST'])
def binary_train():
    if request.method=="POST":
        try:        
            conn,cursor= DBoperator().Create_table()
            trainModelObj = binaryClassification.BinaryModel()
            dataframe,predictions,actual=trainModelObj.training()
            dataframe.to_csv('best_model_summary/binary.csv',index=False)
            conn,data= DBoperator().Insert_table(conn,cursor,actual,predictions)
            DBoperator().fetch_to_csv(data,conn)
        except:
            raise Exception()
        return render_template('result1.html')


@app.route("/multi_train", methods=['POST'])
def multi_train():
    if request.method=='POST':
        try:
            conn,cursor= DBoperator().Create_table() 
            trainModelObj = multiClassification.MultiModel()
            dataframe,predictions,actual=trainModelObj.training()
            dataframe.to_csv('best_model_summary/multi.csv',index=False)
            conn,data= DBoperator().Insert_table(conn,cursor,actual,predictions)
            DBoperator().fetch_to_csv(data,conn)
        except:
            raise Exception()
        return render_template('result2.html')

if __name__ == "__main__":
    app.run(debug=True)