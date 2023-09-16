from flask import Flask
import requests
from flask import render_template
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'
db=SQLAlchemy(app)

class Todo(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    content=db.Column(db.String(200),nullable=False)
    date_created=db.Column(db.DateTime,default=datetime.utcnow)

    def __repr__(self):
        return self.id;

@app.route('/',methods=['POST','GET'])
def index():
    return  render_template('index.html')
# if __name__=="__main__":
#     app.run(debug=True)
    
@app.route('/about' ,methods=['POST','GET'])
def about():
    return render_template('about.html')
# if __name__=="__main__":
#     app.run(debug=True)

@app.route('/mohit')
def mohit():
    response = requests.get('https://en.wikipedia.org/wiki/Hat')
    
    data = json.loads(response.text)
    return str(data)    
    
if __name__=="__main__":
    app.run(debug=True)

    
