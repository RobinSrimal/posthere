from flask import Flask, jsonify, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
from flask_sqlalchemy import SQLAlchemy
import os




app = Flask(__name__) 

# loading the model and the vectorizer
model = joblib.load("post_here_model.joblib")
vectorizer = joblib.load("post_here_vectorizer.joblib")

# creating the subreddit list to link the prediction 
df = pd.read_csv("post_here_subreddits.csv")
subreddit_list = df.subreddit.values.tolist()


APP = Flask(__name__) 
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
DB = SQLAlchemy(APP)




class Record(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    incoming_title = DB.Column(DB.String(100))
    predicted_subreddit = DB.Column(DB.String(100))

    def __repr__(self):
        return '<Provided title {}>'.format(self.incoming_title) + ' <Predicted Subreddit {}>'.format(self.predicted_subreddit)


@app.route('/') 
def hello_world():
    return "welcome to the subreddit predictions api, go to /api"


@app.route('/api', methods=['POST']) 
def make_predict():

    # read in data
    lines = request.get_json(force=True)
    text = lines["body"]
    # transform input string to vector
    vectorized_text = vectorizer.transform([text])
    # use kneighbor to predict fitting subreddit for input string
    prediction = model.kneighbors(vectorized_text, return_distance=False)
    # use the prediction as an index for the subreddit_list
    reddit_prediction = subreddit_list[prediction[0][0]]
    prediction = Record(incoming_title=text, predicted_subreddit=reddit_prediction)
    DB.session.add(prediction)
    DB.session.commit()

    return {"title": reddit_prediction}



if __name__ == '__main__':
    app.run(port = 8080, debug = True)


