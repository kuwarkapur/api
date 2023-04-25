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

def preprocess(url,methods=['GET']):
    
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
@app.route("/predict/<path:url>",methods=['GET'])
def predict(url):
    result = preprocess(url)
    doc=collection.find_one({'Category':str(result).lower().lstrip()})
    del(doc['_id'])
     
    return doc

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        url = request.json["url"]
        result = preprocess(url)
        doc=collection.find_one({'Category':str(result).lower().lstrip()})
        del(doc['_id'])
     
        return doc



@app.route("/food/<string:name>", methods=["GET", "PUT", "DELETE"])
def food(name):
    if request.method == "GET":
        doc = collection.find_one({'Category': name})
        if doc is not None:
            del(doc['_id'])
            return jsonify(doc)
        else:
            return jsonify({'error': 'Food item not found.'}), 404

    elif request.method == "PUT":
        data = request.get_json()
        doc = collection.find_one({'Category': name})
        if doc is not None:
            collection.update_one({'Category': name}, {'$set': data})
            return jsonify({'message': 'Food item updated.'})
        else:
            return jsonify({'error': 'Food item not found.'}), 404

    elif request.method == "DELETE":
        doc = collection.find_one({'Category': name})
        if doc is not None:
            collection.delete_one({'Category': name})
            return jsonify({'message': 'Food item deleted.'})
        else:
            return jsonify({'error': 'Food item not found.'}), 404
                
                
# Define the custom error handler for 404 Error
@app.errorhandler(404)
def not_found_error(error):
    return jsonify(error=str(error)), 404


# Define the custom error handler for 503 errors
@app.errorhandler(503)
def handle_503_error(error):
    return jsonify(error=str(error)), 503


# Define the custom error handler for 500 errors
@app.errorhandler(500)
def handle_500_error(error):
    return jsonify(error=str(error)), 500    
		
            

        
        


if __name__ == '__main__':
    app.run(debug=True)






    
