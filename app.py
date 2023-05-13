import tensorflow as tf
from flask import Flask,request,jsonify
import requests
from flask_cors import CORS, cross_origin
from imageio import imread
import io
import time
import pickle
import cv2
import base64, random, json
from pymongo import MongoClient
import numpy as np
from datetime import datetime

# Create a Flask app instance
PRIVATE_REPL_LINK = 'https:/#########################################/'  # Set the URL of a private REPL (Optional). This URL was found in "Toggle Developers Tool" in "Webview" section inside "Resources" tab. Scroll down to find it with "https://*.id.repl.co/".



MONGO_URI = "mongodb+srv://###################################################/"

# Connect to the MongoDB cluster
client = pymongo.MongoClient(MONGO_URI, connect=False)

# Set the name of the database with its collections
MONGO_DATABASE = "platform"
FEED_COLLECTION = "feed"
USER_COLLECTION = "user"

# Get the database
dbs = client[MONGO_DATABASE]

# Get the collection: Feed & User
feed = dbs[FEED_COLLECTION]
user = dbs[USER_COLLECTION]



 
client = MongoClient("mongodb+srv://###############################")
labels = pickle.load(open('labels.pkl', 'rb'))
dbname= client['database1']
collection=dbname['finalf']

val=tf.keras.models.load_model('Final.h5')
vala=tf.keras.models.load_model('A.h5')
valb=tf.keras.models.load_model('B.h5')
valc=tf.keras.models.load_model('C.h5')
valf=tf.keras.models.load_model('F.h5')
valgg=tf.keras.models.load_model('D.h5')
vale=tf.keras.models.load_model('E.h5')
valg=tf.keras.models.load_model('G.h5')
valh=tf.keras.models.load_model('H.h5')
vali=tf.keras.models.load_model('I.h5')

app = Flask(__name__)
#CORS(app,resources={r"/api/*":{"origins":"*"}})
CORS(app)
#app.config['CORS_HEADERS']='Content-Type'

def fnf(url):
    img = imread(url)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


    interpreter = tf.lite.Interpreter(model_path='fnf.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()


    input_shape = input_details[0]['shape']

    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
    input_tensor= np.array(np.expand_dims(img,0),dtype=np.float32)
    train=['Food', 'NonFood']

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    name= train[pred.argmax()]
    return name
  

def prep(url):

    img = imread(url)
    img=tf.image.resize(img,[128,128])
    img= tf.expand_dims(img,axis=0)
    c= val.predict(img)
    c=labels['Train'][c.argmax()]

    if c=='A':
      list=[]
      c=vala.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t1'][c.argmax()]
      for i in best_n.tolist():
        list.append(labels['t1'][i].lower().lstrip())
        print(list)
    elif c=='B':
      list=[]
      c=valb.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t2'][c.argmax()]
      for i in best_n.tolist():
        list.append(labels['t2'][i].lower().lstrip())
    elif c=='C':
      list=[]
      c=valc.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t3'][c.argmax()]

      for i in best_n.tolist():

        list.append(labels['t3'][i].lower().lstrip())

    elif c=='F':
      list=[]
      c=valf.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t5'][c.argmax()]
      for i in best_n.tolist():
        list.append(labels['t5'][i].lower().lstrip())
    elif c=='D':
      list=[]
      c=valgg.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t4'][c.argmax()]
      for i in best_n.tolist():
        list.append(labels['t4'][i].lower().lstrip())
    elif c=='E':
      list=[]
      c=vale.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t9'][c.argmax()]
      for i in best_n.tolist():
        list.append(labels['t9'][i].lower().lstrip())
    elif c=='G':
      list=[]
      c=valg.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t6'][c.argmax()]
      for i in best_n.tolist():
        list.append(labels['t6'][i].lower().lstrip())
    elif c=='H':
      list=[]
      c=valh.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t7'][c.argmax()]
      for i in best_n.tolist():
        list.append(labels['t7'][i].lower().lstrip())
    else:
      list=[]
      c=vali.predict(img)
      best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      c=labels['t8'][c.argmax()]
      #best_n = np.argsort(-c, axis=1)[:, :10].reshape(-1)
      for i in best_n.tolist():
        list.append(labels['t8'][i].lower().lstrip())
    
    return c,list

def preprocess(base64_str):

    print(base64_str)
    img = imread(io.BytesIO(base64.b64decode(base64_str)))
    #img = imread(url)


    interpreter = tf.lite.Interpreter(model_path='tflite_mod.tflite')
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




def updateMongoDocumentUser(uuid, query, collection):
    '''
	Function updateMongoDocumentUser(uuid, query, collection):
	This function updates a user's document in MongoDB by adding a new post to their posts_array.

	Parameters:
	uuid (str): A string representing the user's unique identifier in the database.
	query (dict): A dictionary representing the new post that will be added to the user's posts_array.
	collection (pymongo.collection.Collection): A MongoDB collection object representing the user's collection.
	Returns:
	update (dict): A dictionary representing the update operation that was performed on the user's document.
	list(collection.find({"user_google_uuid": uuid})) (list): A list of dictionaries representing all documents in the user's collection that match the given UUID.
	'''
    update = {"$push": {'posts_array': query}}
    user.update_one({"user_google_uuid": uuid}, update)

    return update, list(collection.find({"user_google_uuid": uuid}))


def add_feed_function(json):
    '''
	Function add_feed_function(json):
	This function adds a new post to a feed collection in MongoDB.
	
	Parameters:
	json (dict): A dictionary representing the new post that will be added to the feed collection.
	Returns:
	{"status": 200, "msg": "Sucessfully created post."} (dict): A dictionary indicating that the post was successfully created.
	'''

    inserted_id = feed.insert_one(json).inserted_id
    print(json, inserted_id)

    return {"status": 200, "msg": "Sucessfully created post."}


# Function to delete a document from the user collection by serial number (sno)
def deleteMongoDocumentUser(sno, collection):

    # Delete the document with the specified sno from the collection
    user.delete_one({"sno": sno})

    # Return a message indicating that the document has been deleted for the specified sno
    return "Deleted user for <sno>: " + str(sno)


# Function to delete a document from the feed collection by serial number (sno)
def deleteMongoDocumentFeed(sno, collection):

    # Delete the document with the specified sno from the collection
    feed.delete_one({"sno": sno})

    # Return a message indicating that the document has been deleted for the specified sno
    return "Deleted feed for <sno>: " + str(sno)


# Define the User class
class User:
    '''
	User: A class that represents a user object with various attributes such as user name, email, etc.
	It has a method called generate_document_JSON that returns a JSON document representing the user object.
	'''
    def __init__(
        self,
        user_name,
        user_google_uuid,
        user_email,
        total_posts,
        user_total_health,
        posts_array,
        sno,
    ):
        # Initialize the attributes of the User object
        self.user_name = user_name
        self.user_google_uuid = user_google_uuid
        self.user_email = user_email
        self.total_posts = total_posts
        self.user_total_health = user_total_health
        self.posts_array = posts_array
        self.sno = sno

# Method to generate a JSON document representing the User object

    def generate_document_JSON(self):
        return {
            "user_name": self.user_name,
            "user_google_uuid": self.user_google_uuid,
            "user_email": self.user_email,
            "total_posts": self.total_posts,
            "user_total_health": self.user_total_health,
            "posts_array": self.posts_array,
            "sno": self.sno,
        }


# Define the Feed class
class Feed:
    '''
	Feed: A class that represents a feed object with attributes such as food name, health index, etc.
 	It has a method called generate_document_JSON that returns a JSON document representing the feed object.
	'''
    def __init__(
        self,
        uuid,
        user_name,
        post_id,
        image_url,
        health_index,
        food_name,
        carbs,
        fat,
        cal,
        protein,
        sugar,
        sno,
    ):
        # Initialize the attributes of the Feed object
        self.user_name = user_name
        self.uuid = uuid
        self.post_id = post_id
        self.image_url = image_url
        self.health_index = health_index
        self.food_name = food_name
        self.carbs = carbs
        self.fat = fat
        self.cal = cal
        self.protein = protein
        self.sugar = sugar
        self.sno = sno

# Method to generate a JSON document representing the Feed object

    def generate_document_JSON(self):
        return {
            "user_name": self.user_name,
            "uuid": self.uuid,
            "post_id": self.post_id,
            "image_url": self.image_url,
            "health_index": self.health_index,
            "food_name": self.food_name,
            "carbs": self.carbs,
            "fat": self.fat,
            "cal": self.cal,
            "protein": self.protein,
            "sugar": self.sugar,
            "sno": self.sno,
        }


@app.route("/pred/<value>")
@cross_origin()
def pred(value):
    doc=collection.find_one({'item_name':str(value).lower().lstrip()})
    if doc is None:
        return {'status':503, 'msg': "not found in database", 'display_msg': "Regret to inform you that this is not present in our database right now"}
    else:
        del(doc['_id'])
        doc.update({'status':200})
        doc.update({'msg': "found in database"})
        doc.update({'display_msg': ""})

    return doc

@app.route("/predict/<path:url>")
@cross_origin()
def predict(url):
    res=fnf(url)
    print(res)
    if res=='Food':
        result,top_10 = prep(url)
        if result is None:
            return {'status':503, 'msg': "ml model not able to detect", 'display_msg': "click a better picture"}
        doc=collection.find_one({'item_name':str(result).lower().lstrip()})
        if doc is None:
            return {'status':503, 'msg': "not found in database", 'display_msg': "Regret to inform you that this is not present in our database right now"}
        else:
            print('sucess')
            del(doc['_id'])
            doc.update({'status':200})
            doc.update({'msg': "working well"})
            doc.update({'display_msg': ""})
            doc.update({'name':top_10})
        #ref = db.reference("/")
        #ref.child(value).set(str(doc))
            print(doc)
    else:
        print('nf')
        return {'status':400, 'msg': "not a picture of food", 'display_msg': "Please click a picture of food"}

    return doc

@app.route('/predictbase64',methods=["POST"])
@cross_origin()
def predictbase64():
    if request.method=='POST':
        data = request.form
        base64_str = data.getlist('base64_str')[0]

    result = str(preprocess(base64_str))
    print(result)
    doc=collection.find_one({'item name':result.lower()})
    print(doc)
    if doc is None:
        return jsonify('click a better picture')
    else:
        print('sucess')
        del(doc['_id'])
        ref = db.reference("/")
        ref.child(value).set(str(doc))
        print(doc)
        return doc
                
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






    
