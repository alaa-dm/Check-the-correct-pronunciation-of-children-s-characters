from python_speech_features import mfcc
import os
import librosa
from joblib import dump, load
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

clf = load('filename1.joblib')

path='C:/Users/USER/Desktop/data2'
lables=os.listdir(path)
wav1=[]
for lable in lables :
     data, sampling_rate = librosa.load(path + '/' + lable)
     (rate, sig) = wav.read(path + '/' + lable)
     var_mfcc =mfcc(sig, rate)
     var_mean_mfcc=var_mfcc.mean(axis=0)
     wav1.append(var_mean_mfcc)
x_test=wav1

y=clf.predict(x_test)
print(y)
plot_confusion_matrix(clf,x_test,y)
plt.show()
# acc = clf.score(normalized_arr,y)
# print("acc=%0.3f" %acc)