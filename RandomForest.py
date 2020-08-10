
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#########################################
#########################################
#### Loading and splitting the data #####



#this opens the file with inputs
data = pd.read_csv('RandomForest.csv')
data.head()

training_size = .7
split = int(len(data)*.7)


X=data[['Open','High','Low','Close','Volume','Accumulation Distribution Line','MACD','Chaikan Oscillator (CHO)','Highest closing price (5 days)',
'Lowest closing price (days)','Stochastic %K (5 days)','%D','Volume Price Trend (VPT)','Williams %R (14 days)','Relative Strength Index','Momentum (10 days)',
'Price rate of change (PROC)','Volume rate of change (VROC)','On Balance Volume (OBV)']]
y=data['Outputs']


#splitting up the dataset
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]




#create a gaussian classifier
clf=RandomForestClassifier(n_estimators=200) #number of trees

#train the model on the training data
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

