from flask import Flask
from flask import render_template
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import torch
app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'
db=SQLAlchemy(app)

class Todo(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    content=db.Column(db.String(200),nullable=False)
    date_created=db.Column(db.DateTime,default=datetime.utcnow)

    def __repr__(self):
        return self.id;

with open("classes.txt","r") as file:
    class_names = file.read()
    class_names = eval(class_names)




model = torch.load("Image_model.pt",map_location=torch.device('cpu'))
model.eval()
from Image.going_modular.going_modular.predictions import pred_and_plot_image

# Setup custom image path
custom_image_path = "PotatoHealthy1.JPG"
actual_class=["Healty_potato"]



@app.route('/',methods=['POST','GET'])
def index():
    # Predict on custom image
    x = pred_and_plot_image(model=model,
                        image_path=custom_image_path,
                        class_names=class_names)

    return json.dumps({"predicted_class":x[0],"with_probabilty":x[1]})
    
    
if __name__=="__main__":
    app.run(debug=True)
    
