#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import joblib


# In[2]:


app = Flask(__name__)
model = joblib.load(open('text_classifier.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    try:

        int_features = [x for x in request.form.values()]
        X_test=[]
        X_test.append(int_features[0])
        prediction = model.predict(X_test).tolist()[0]
        probas = model.predict_proba(X_test).tolist()[0]
        pred_json = json.dumps({'class_predicted': prediction, 'predict_proba':probas})
        
        output = pred_json
        return render_template('index.html', Suggestion_text=output)

    except:
        return render_template('index.html', Suggestion_text="Wrong Input, Try again")
    
    
@app.route('/predict_api', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        try:

            data = json.loads(request.data)
            
            int_features = data.get('item_cat')
            X_test=[]
            X_test.append(int_features[0])
            prediction = model.predict(X_test).tolist()[0]
            probas = model.predict_proba(X_test).tolist()[0]
            pred_json = json.dumps({'class_predicted': prediction, 'predict_proba':probas})



        except:
            output = "Wrong Input, Try again"
            result1 = {
                "input error": output
            }

        return jsonify(pred_json)


if __name__ == "__main__":
    app.run()


# In[ ]:




