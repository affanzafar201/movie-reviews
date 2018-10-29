import pickle
import tensorflow as tf
import numpy as np
from flask import request, url_for, jsonify, Flask
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

def transform_data(sentence):
    try:
        input_dim = 150
        sentence = [sentence]
        sentence = [' '.join(x.split()[:input_dim]) for x in sentence]
        sentence = np.array(sentence, dtype=object)
        print(sentence)
    except:
        print("Unable to hold")
    # loading tokenizer
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tok = pickle.load(handle)
            sentence_matrix = tok.texts_to_sequences(sentence)
            print(sentence_matrix)
            sentence_matrix = pad_sequences(sentence_matrix, maxlen=input_dim, padding='post', truncating='post', value=0)
            return sentence_matrix
    except:
        print("unable to load tokenizer")

def load_mdl():
    # Loading model
    global mdl,graph
    mdl = load_model('models/dl_arch.h5')
    graph = tf.get_default_graph()


def predict_sentiment(review):
    response = dict()
    try:
        test_matrix = transform_data(review)
    except:
        print("unable to transform data")
        return
    # loading model   
    print(mdl.summary())
    response['movie_review'] = review
    with graph.as_default():
        response['prob']  = str(mdl.predict(test_matrix)[0][0])
        if float(response['prob'])>=0.5:
            response['sentiment'] = "Good"
        else:
            response['sentiment'] = "Bad"

        print(response)
    
        return response


@app.route('/test/')
def example():
    return {'hello': 'world'}




@app.route('/predict',methods=['POST'])
def predict(): 
    try:
        json_data = request.get_json()
        print(request.get_json())
        movie_name = json_data.get('movie_name')
        movie_review = json_data.get('movie_review')
    
    except:
        p = dict()
        p['error'] = "No valid json provided"
        return jsonify(p)


    return jsonify(predict_sentiment(movie_review))


if __name__=='__main__':

    load_mdl()
    app.run(debug=False,threaded=True)
