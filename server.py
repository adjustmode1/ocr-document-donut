from flask import Response, Flask, render_template, request, redirect, url_for, jsonify, make_response
import cv2
import numpy as np
import os
import base64
import uuid
import json
from flask_cors import CORS
import random
import string

app = Flask(__name__)
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')
# Define the route for the index page

if __name__ == '__main__':
    app.secret_key = 'mysecretkey'
    load_faces()
    app.run(debug=True)