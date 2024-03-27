# -*- coding: utf-8 -*-
"""utsaksa3ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rOun0sVyhb9G9LobHd6GcvtaWnC0XIE-
"""

import numpy as np # linear algebra
import pandas as pd

filePath = 'https://raw.githubusercontent.com/aksajr/aksa/main/heart.csv'

# Baca file CSV ke dalam dataframe
df = pd.read_csv(path)

#
df.head()

df.info()

#menetapkan variabel
num = ['age' , 'sex', 'cp', 'trestbps', 'chol',  'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df[num].describe()

df.corr()

# Membuat plot countplot menggunakan Seaborn
sns.countplot(data=df, x="age", hue="target")

df.head()

x = df.drop(['target'], axis=1)  # Variabel Numerik (Independen)
y = df['target']  # Variabel Dependen

from sklearn.model_selection import train_test_split

#Split data set menjadi 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#check x_train and x_test
x_train.shape, x_test.shape

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.model_selection import cross_val_score

#membuat model logistic regression
logreg = LogisticRegression()

# melatih model
logreg.fit(x_train, y_train)

# membuat variabel prediksi
y_pred = logreg.predict(x_test)

# menghitung nilai accuracy model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression model:", accuracy)

from sklearn.ensemble import RandomForestClassifier

# membuat model random forest classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# melatih model
y_pred = rfc.predict(x_test)

# membuat variabel prediksi
accuracy = accuracy_score(y_test, y_pred)

# menghitung nilai accuracy model
print("Accuracy on the test set:", accuracy)

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing

Accuracies= {} # membuat akulasi untuk menyimpan model akurasi

# mendefinisikan model
LR = LogisticRegression()

# hyperparameters
parameters=[{'penalty':['l2'],'C':[0.1,0.4,0.5],'random_state':[0]}]

# mendefinisikan search dengan cross validation
search = GridSearchCV(LR, parameters, scoring='accuracy', n_jobs=-1)

# hasil
result = search.fit(x_train, y_train)
# hasil
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# mendefinisikan model
Rfc=RandomForestClassifier()

# hyperparameters
parameters = [{'max_depth': np.arange(5, 10),'min_samples_split': np.arange(2, 3),
               'n_estimators': np.arange(10, 20), 'criterion': ["gini", "entropy"]}]

# mendefinisikan search dengan cross validation
search = GridSearchCV(Rfc, parameters, scoring='accuracy', n_jobs=-1)

# hasil
result = search.fit(x_train, y_train)

# hasil
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
model = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu'),
tf.keras.layers.Dense(16, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.BinaryCrossentropy(),
metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=25, validation_data=(x_test, y_test))

df.head()

# Assuming you have already defined and fitted the scaler object named 'scaler'

import numpy as np

# Define the new summary
new_summary = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])

# Scale the new summary using the scaler
new_summary_scaled = scaler.transform(new_summary)

# Prediksi dengan menggunakan madel yang telah dilatih
predictions = model.predict(new_summary_scaled)

# konversi ke yes atau no dengan  threshold (0.5)
binary_predictions = (predictions > 0.5).astype(int)

#
print(f"Predicted Probability: {predictions[0][0]}")
print(f"Binary Prediction: {binary_predictions[0][0]}")
print()

if (binary_predictions==0):
    print("Sakit Jantung...")
else:
    print("Tidak Sakit Jantung")

print()
