from flask import Flask, make_response, request, render_template, session, url_for, redirect, send_from_directory, send_file
from pandas import DataFrame
import io
from datetime import datetime, date, timedelta
from io import StringIO
import csv
import pandas as pd
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


 


app = Flask(__name__)
#clear excel file cache
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def form():
    return render_template('index.html', data=[6, 12, 24])



def create_dataset(dataset, look_back=1):
            data = []
            for i in range(len(dataset)-look_back-1):
                    a = dataset[i:(i+look_back), 0]
                    data.append(a)
            return np.array(data)
        
        
   

@app.route('/transform', methods=["POST"])
def transform_view():
 year = [12, 24 ,36]   
 if request.method == 'POST':
        f = request.files['data_file']
        if not f:
            return redirect(url_for("form")) 

        #read file data
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        stream.seek(0)
        result = stream.read()
        df = pd.read_csv(StringIO(result), usecols=[1])
        dataset = df.values
        dataset = dataset.astype('float32')
        
        #scale data into particular range
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        
        #look back 1 month ago
        look_back = 1
        dataset_look = create_dataset(dataset, look_back)
        dataset_look = np.reshape(dataset_look, (dataset_look.shape[0], 1, dataset_look.shape[1]))
        
        # load the model from disk
        model = load_model('model.h5')
        predict = model.predict(dataset_look,1,200,100)
        
        #transform data after predict
        transform = scaler.inverse_transform(predict)
        
        #Predict future value
        X_FUTURE = int(request.form['comp_select'])
        transform = np.array([])
        last = dataset[-1]
        for i in range(X_FUTURE):
            curr_prediction = model.predict(np.array([last]).reshape(1, look_back, 1))
            last = np.concatenate([last[1:], curr_prediction.reshape(-1)])
            transform = np.concatenate([transform, curr_prediction[0]])
      
        transform = scaler.inverse_transform([transform])[0]

        #extract month value
        df2 = pd.read_csv(StringIO(result))
        matrix2 = df2[df2.columns[0]].to_numpy()
        list1 = matrix2.tolist()

        #combine month and predicted value
        dicts = []
        curr_date = pd.to_datetime(list1[-1])
        for i in range(X_FUTURE):
            curr_date = curr_date +  relativedelta(months=+1)
            dicts.append({'Predictions': transform[i], "Month": curr_date})
            
        #make dataframe and store to a new CSV file
        new_data = pd.DataFrame(dicts).set_index("Month")
        new_data.to_csv(os.path.join("downloads","result.csv"), index = True, encoding='utf-8')

       
        #label X and Y value
        labels = [datetime.datetime.strftime(d['Month'], "%d-%m-%Y") for d in dicts]
        values = [d['Predictions'] for d in dicts]
        colors = [ "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
                   "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
                   "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

        line_labels=labels
        line_values=values
        return render_template('graph.html', title='Time Series Sales forecasting', max= (max(values)+ max(values)), labels=line_labels, values=line_values)
    
        ##df_predict = pd.DataFrame(transform, columns=["predicted value"])
        ##response = make_response(new_data.to_csv(index = True, encoding='utf8'))
        ##response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        ##return response

@app.route('/download')
def download():
    path = "downloads/result.csv"
    return send_file(path , as_attachment = True)     



if __name__ == "__main__":
    app.run(debug=True, port = 9000, host = "localhost")
