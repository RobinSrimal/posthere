from flask import Flask


APP = Flask(__name__) 



@app.route('/')
def hello_world():
    return 'Hello datonistas'