# Import the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns  # Add import statement for seaborn

# Assuming the file path is correct
filePath = 'https://raw.githubusercontent.com/aksajr/aksa/main/heart.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(filePath)

# Display the first few rows of the DataFrame
df.head()

# Display information about the DataFrame
df.info()

# Define the numeric columns
num = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Display descriptive statistics for the numeric columns
df[num].describe()

# Calculate the correlation matrix
df.corr()

# Create a countplot using Seaborn
sns.countplot(data=df, x="age", hue="target")

# Display the first few rows of the DataFrame
df.head()

# Define the independent (x) and dependent (y) variables
x = df.drop(['target'], axis=1)  # Independent variables
y = df['target']  # Dependent variable

from sklearn.model_selection import train_test_split

#Split data set menjadi 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train.shape, x_test.shape

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

#membuat model logistic regression
logreg = LogisticRegression()

# melatih model
logreg.fit(x_train, y_train)

# membuat variabel prediksi
y_pred = logreg.predict(x_test)

# menghitung nilai accuracy model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression model:", accuracy)
