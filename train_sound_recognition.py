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
     var_mfcc =mfcc(sig, rate)
     var_mean_mfcc=var_mfcc.mean(axis=0)
     all_wave.append(var_mean_mfcc)
     all_lable.append(lable)
print(all_lable)


# numpy array all_wave
all_waves=np.array(all_wave)
print(all_waves.shape)
X=all_waves

# normalizetion all_wave
# normalized_arr = preprocessing.normalize(all_waves,axis=0)
# print(normalized_arr.shape)


# Load data from numpy file
# X =  normalized_arr
y =  all_lable
# instantiating the random over sampler

ros = RandomOverSampler()
# resampling X, y
X_ros, y_ros = ros.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_ros))

# # Split data into training and test subsets
X_train,X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.3, random_state=0)

# Simple SVM
print('fitting...')
clf = SVC(C=400 ,gamma=0.01)
svc=clf.fit(X_train,y_train)
acc = clf.score(X_test, y_test)
print("acc=%0.3f" %acc)
joblib.dump(clf,'filename1.joblib')

y_pred = clf.predict(X_test)

# confusion_matrix
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test,y_pred,)
print(cf_matrix)


# plot confusion matrix
plot_confusion_matrix(clf,X_test,y_test)
plot_confusion_matrix(clf,X_test,y_pred)
plt.show()


# Simple logistic
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("acc=%0.3f" %score)

# Simple knn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=60)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred )
