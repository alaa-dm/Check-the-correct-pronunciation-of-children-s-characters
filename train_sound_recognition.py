import os
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import numpy as shape
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import joblib
from joblib import dump, load
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import random


# load data`
path='c:/Users/USER/Desktop/data'
lables=os.listdir(path)
all_wave=[]
all_lable=[]



# Features Extraction
for lable in lables :
    ws=[f for f in os.listdir(path + '/' + lable)]
    for w in ws:
     data, sampling_rate = librosa.load(path + '/' + lable + '/' + w)
     (rate, sig) = wav.read(path + '/' + lable + '/' + w)
     var_mfcc =mfcc(sig, 16000)
     var_mean_mfcc=var_mfcc.mean(axis=0)
     all_wave.append(var_mean_mfcc)
     all_lable.append(lable)
# print(all_lable)


# numpy array all_wave
all_waves=np.array(all_wave)
#  print(all_waves.shape)
X=  all_waves
y = all_lable

# normalizetion all_wave
# normalized_arr = preprocessing.normalize(all_waves,axis=0)
# print(normalized_arr.shape)
# X =  normalized_arr
# instantiating the random over sampler
# resampling X, y
# random.seed(20)
ros = RandomOverSampler(random_state=50)
X_ros, y_ros = ros.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y))
print('Resampled dataset shape %s' % Counter(y_ros))
# # Split data into training and test subsets
X_train,X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.3, random_state=0)
#
##1- Simple SVM
print('fitting...')
clf = SVC(C=400,gamma=0.001)
svc=clf.fit(X_train,y_train)
acc = clf.score(X_test, y_test)
print("acc=%0.3f" %acc)

joblib.dump(clf,'filename1.joblib')
y_pred = clf.predict(X_test)

# confusion_matrix
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test,y_pred,)
print(cf_matrix)


## plot confusion matrix
plot_confusion_matrix(clf,X_test,y_test)
plot_confusion_matrix(clf,X_test,y_pred)
plt.show()


##2- Simple logistic
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
# print("acc=%0.3f" %score)

##3- Simple knn
# neigh =  KNeighborsClassifier(n_neighbors=50)
# neigh.fit(X_train,y_train)
# acc1 = neigh.score(X_test, y_test)
# print("acc=%0.3f" %acc1)

#4- Simple GaussianProcessClassifie
kernel = 0.5* RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train,y_train)
acc2=gpc.score(X_test, y_test)
print("acc=%0.3f" %acc2)