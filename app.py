import tensorflow as tf
from flask import Flask,request,render_template
from skimage import io
import cv2
import os
from pymongo import MongoClient
import numpy as np
train=['ABALONE',
 'ACEROLA',
 'ELDERBERRIES',
 'ENDIVE',
 'ENGLISH MUFFIN',
 'EPAZOTE',
 'FALAFEL',
 'FEIJOA',
 'FENNEL',
 'FIDDLEHEAD FERNS',
 'FIREWEED',
 'KACHORI',
 'KALE',
 'KAMUT',
 'KANPYO',
 'PILINUTS-CANARYTREE',
 'PILLSBURY GRANDS',
 'PIMENTO',
 'PINE NUTS'] #train is the list of food items 
client = MongoClient("mongodb+srv://############################################")

dbname= client['db']
collection=dbname['foood']


app = Flask(__name__)

def preprocess(url):
    
    img = io.imread(url)

    interpreter = tf.lite.Interpreter(model_path='tflite_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()


    input_shape = input_details[0]['shape']

    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
    input_tensor= np.array(np.expand_dims(img,0),dtype=np.float32)


    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    name= train[pred.argmax()]
    return name

    
    return name
@app.route("/predict/<path:url>")
def predict(url):
    result = preprocess(url)
    doc=collection.find_one({'Category':str(result)})
    del(doc['_id'])
     
    return doc
                
                
    
		
            

        
        


if __name__ == '__main__':
    app.run(debug=True)






    
