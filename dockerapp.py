from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import time
import datetime as dt
from flask import Flask, jsonify
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import logging
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

global fr
cluster = Cluster(contact_points=['192.168.1.107'], port=9042)
session = cluster.connect()
KEYSPACE = "mykeyspace"
app = Flask(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# TensorFlow and tf.keras
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def createKeySpace():
   log.info("Creating keyspace...")
   try:
       session.execute("""
           CREATE KEYSPACE IF NOT EXISTS %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
           """ % KEYSPACE)
       log.info("setting keyspace...")
       session.set_keyspace(KEYSPACE)
       log.info("creating table...")
       session.execute("""
           CREATE TABLE mytable (
               time text,
               name text,
               perdict text,
               PRIMARY KEY (name)
           )
           """)
   except Exception as e:
       log.error("Unable to create keyspace")
       log.error(e)

createKeySpace();

def insertdata(fr,perdict):

    try:
        n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        session.execute("""
            INSERT INTO mytable (time, name, perdict)
            VALUES ('%s', '%s', '%s')
            """ % (n, fr, perdict))
        log.info("%s, %s, %s" % (n, fr, perdict))
        log.info("Data stored!")
        #session.execute("""Select * from mytable;""")
    except Exception as e:
        log.error("Unable to insert data!")
        log.error(e)


def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array, img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    a = class_names[predicted_label]
    return a


@app.route('/file/<file>', methods=['GET'])
def anas(file):
    fr = '/images/' + file
    def analysis(fr):
        model = keras.models.load_model(
            '/model/my_model.h5')
        test_images = Image.open(fr)
        #print(type(test_images))
        test_images = np.invert(test_images.convert('L'))
        test_images = test_images / 255.0
        probability_model = tf.keras.Sequential([model,
                                                 tf.keras.layers.Softmax()])
        test_images = (np.expand_dims(test_images, 0))
        predictions_single = probability_model.predict(test_images)
        b = plot_image(0, predictions_single[0], test_images)
        return b
    perdict = analysis(fr)
    insertdata(fr,perdict)
    return perdict

#anas('pullover1.jpg')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug = True, use_reloader = False)
