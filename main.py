from flask import Flask, request, jsonify ,render_template
import pickle
import json
import numpy as np

app = Flask(__name__)
model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))

def predict_houseprice(location,area,bath,bhk):
    with open("columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(data_columns))
    x[0] = area
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    return round(model.predict([x])[0],2)


@app.route('/')
def home():
    return render_template('newapp.html')

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form['location']
    area = int(request.form['area'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])
    output=predict_houseprice(location,area,bath,bhk)

    return render_template('newapp.html', prediction_text='Predicted House price is : {} lakhs'.format(output))

if __name__ == "__main__":
    app.run(debug=True)