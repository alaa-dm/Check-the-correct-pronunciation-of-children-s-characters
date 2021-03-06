from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from app import __init__
from joblib import dump, load
import scipy.io.wavfile as wav
import librosa
import numpy as np
import urllib.request
from python_speech_features import mfcc
clf = load('filename1.joblib')

app=Flask(__name__,template_folder="templates")
ALLOWED_EXTENSIONS = set(['wav'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = 'D:/upload'

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
@app.route('/')
def upload_file1():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
         f.save(secure_filename(f.filename))
         print(f.filename)
         all_wave = []
         data, sampling_rate = librosa.load(f.filename)
         (rate, sig) = wav.read(f.filename)
         os.remove(f.filename)
         var_mfcc = mfcc(sig, 16000)
         var_mean_mfcc = var_mfcc.mean(axis=0)
         all_wave.append(var_mean_mfcc)
         all_waves = np.array(all_wave)
         x_test = all_waves
         y = clf.predict(x_test)
         # print(y)
         result= y
         return str(result[0])

         return render_template("upload.html", character_name= result)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host="0.0.0.0")