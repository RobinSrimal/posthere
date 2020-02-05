from flask import Flask, jsonify, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib




app = Flask(__name__) 

# loading the model and the vectorizer
model = joblib.load("post_here_model.joblib")
vectorizer = joblib.load("post_here_vectorizer.joblib")

# creating the subreddit list to link the prediction 
df = pd.read_csv("post_here_subreddits.csv")
subreddit_list = df.subreddit.values.tolist()


@app.route('/') 
def hello_world():
    return "welcome to the subreddit predictions api, go to /api"


@app.route('/api', methods=['POST']) 
def make_predict():

    # read in data
    text = request.get_json(force=True)
    # transform input string to vector
    vectorized_text = vectorizer.transform([text])
    # use kneighbor to predict fitting subreddit for input string
    prediction = model.kneighbors(vectorized_text, return_distance=False)
    # use the prediction as an index for the subreddit_list
    reddit_prediction = subreddit_list[prediction[0][0]]
    # put the string into a list to make jsonify happy
    reddit_prediction_in_list =[reddit_prediction]

    return jsonify(title = reddit_prediction_in_list)



if __name__ == '__main__':
    app.run(port = 8080, debug = True)


