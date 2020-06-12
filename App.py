from __future__ import division, print_function
import numpy as np
import os
from tensorflow.python.keras.backend import set_session
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
import webbrowser
import time

sess = tf.Session()
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

set_session(sess)
model = load_model("Preprocessed and Model/garbage.h5")

print('Model loaded Check http://localhost:5000/')

@app.route('/',methods =['GET'])
def index( ):
    return render_template('base.html')

time.sleep(5)    
webbrowser.open_new("http://localhost:5000/")
 
@app.route('/predict',methods = ['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        print("upload folder is ", file_path)
        f.save(file_path)
        
        img = image.load_img(file_path,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            set_session(sess)
            preds = model.predict_classes(x)          
            print("prediction",preds)
            
        index = ['Cardboard','Glass','Metal','Paper','Plastic','Trash']
        
        text = "The predicted garbage is : " + str(index[preds[0]])
        
    return text

if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    