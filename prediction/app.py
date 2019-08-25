from flask import Flask

application = Flask(__name__)

from flask import request
from flask import render_template
from flask import flash

import json
import os
import sys
import numpy as np
from numpy import array
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class Config(object):
  SECRET_KEY = '828sdfhjsdffre8f'

class QueryForm(FlaskForm):
  query = StringField('Enter words here, space delimited:', validators=[DataRequired()])
  submit = SubmitField('Submit')

application.config.from_object(Config)

sess = tf.Session()
set_session(sess)
model = load_model('model/mortgage_doc_mlp_model.h5')
graph = tf.get_default_graph()

pickle_in = open('pickles/label_encoder.pickle','rb')
label_encoder = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('pickles/vectorizer.pickle','rb')
vectorizer = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('pickles/selector.pickle','rb')
selector = pickle.load(pickle_in)
pickle_in.close()

   
@application.route('/',methods=['GET','POST'])
@application.route('/index',methods=['GET','POST'])
def index():
    form = QueryForm()

    if form.validate_on_submit():
        prediction_class, prediction_proba = get_prediction(form.query.data)
        flash("Predicted Class: {}, Probability: {}".format(prediction_class, prediction_proba))
    return render_template('index.html',
                           form = QueryForm())

@application.route('/predict',methods=['GET'])
def predict():  
    words = request.args.get('words')
    if words:
        prediction_class, prediction_proba = get_prediction(words)
    else:
        prediction_class = ['No words sent from which to make prediction']
        prediction_proba = "0"

    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": prediction_class,
            "confidence": str(prediction_proba)
        })
    }

def get_prediction(words):
    words = array([words])      
    words_vector = vectorizer.transform(words)
    words_vector = selector.transform(words_vector).astype('float32')
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        prediction = model.predict_proba(words_vector)
        prediction_class_encoded = np.argmax(prediction)
        prediction_proba = np.amax(prediction)
        prediction_class_decoded = label_encoder.inverse_transform([prediction_class_encoded]).tolist()
    return prediction_class_decoded, prediction_proba


if __name__ == "__main__":
    application.run()
