from flask import Flask
from flask import request, redirect, url_for
from flask import render_template
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import os
import torch
from flask import Flask, render_template, request, jsonify
import joblib
app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'
db=SQLAlchemy(app)

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
def handle_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return 'Invalid file type'

        upload_folder = 'E:/flask/plant_disease_identification/uploads'

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Process the file or perform any necessary actions
        redirect_url = url_for('predict',file_path=file_path)
        return redirect(redirect_url)
    

        
    
        
    
    # If it's a GET request, render an HTML form for file upload
    return render_template('upload.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    file_path = request.args.get('file_path')
    custom_image_path = file_path
    x = pred_and_plot_image(model=model,
                            image_path=custom_image_path,
                            class_names=class_names)
    
    os.remove(file_path)
    return json.dumps({"predicted_class":x[0],"with_probabilty":x[1]})



if __name__=="__main__":
    app.run(debug=True)
    
