from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
model = pickle.load(open('pay.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    features=['Crop_Year', 'Rain_Fall', 'Temperature', 'Area', 'Humidity', 'N', 'P',
       'K', 'ANANTAPUR', 'CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'KADAPA',
       'KRISHNA', 'KURNOOL', 'PRAKASAM', 'SPSR NELLORE', 'SRIKAKULAM',
       'VISAKHAPATANAM', 'VIZIANAGARAM', 'WEST GODAVARI', 'Autumn', 'Kharif',
       'Rabi', 'Summer', 'Whole Year', 'Winter', 'Arecanut', 'Arhar/Tur',
       'Bajra', 'Banana', 'Beans & Mutter(Vegetable)', 'Bhindi',
       'Bottle Gourd', 'Brinjal', 'Cabbage', 'Cashewnut', 'Castor seed',
       'Citrus Fruit', 'Coconut ', 'Coriander', 'Cotton(lint)',
       'Cowpea(Lobia)', 'Cucumber', 'Dry chillies', 'Dry ginger', 'Garlic',
       'Ginger', 'Gram', 'Grapes', 'Groundnut', 'Horse-gram', 'Jowar', 'Korra',
       'Lemon', 'Linseed', 'Maize', 'Mango', 'Masoor', 'Mesta',
       'Moong(Green Gram)', 'Niger seed', 'Onion', 'Orange',
       'Other  Rabi pulses', 'Other Fresh Fruits', 'Other Kharif pulses',
       'Other Vegetables', 'Papaya', 'Peas  (vegetable)', 'Pome Fruit',
       'Pome Granet', 'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice',
       'Safflower', 'Samai', 'Sannhamp', 'Sapota', 'Sesamum', 'Small millets',
       'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 'Tapioca',
       'Tobacco', 'Tomato', 'Turmeric', 'Urad', 'Varagu', 'Wheat',
       'other fibres', 'other misc. pulses', 'other oilseeds',
       'Andhra Pradesh']
    l=[0 for i in range(len(features))]
    xp=pd.DataFrame(columns=features)
    xp.loc[0]=l
    xp['Andhra Pradesh']=1
    xp['N']= int(request.form['Nitrogen'])
    xp['P']= int(request.form['Phosporus'])
    xp['K']= int(request.form['Potassium'])
    xp['Temperature']= int(request.form['Temperature'])
    xp['Humidity']= int(request.form['Humidity'])
    xp['Crop_Year']= int(request.form['crop_year'])
    xp['Rain_Fall']= int(request.form['Rain_fall'])
    xp['Area']= int(request.form['Area'])
    dst= request.form['district']
    dst=dst.upper()
    xp[dst]=1
    crop_name= request.form['crop_name']
    xp[crop_name]=1
    season= request.form['season']
    xp[season]=1

    p=xp[0:1]

    result=model.predict(p)
    result=result[0]

    return render_template('after.html', data=result)

if __name__ == "__main__":
    app.run(debug=True)
