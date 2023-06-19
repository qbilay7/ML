///
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


dataset=pd.read_csv('Desktop/data.csv')
print(dataset)
#Info
print(dataset.head())
print(dataset.info())
print(dataset.describe())
dataset=dataset.dropna()  #To get rid of empty values

#Visualization:
data=zip(dataset['danceability'], dataset['liveness'], dataset['liked'])
syn_data=pd.DataFrame(data, columns=['danceability', 'liveness', 'liked'])
print(syn_data.head())

plt.plot(syn_data['danceability'],syn_data['liveness'])
sns.lineplot(x='danceability', y='liveness', data=syn_data) #Line Plot
plt.show()

plt.bar(syn_data['danceability'],syn_data['liveness'])
sns.barplot(x='danceability', y='liveness', data=syn_data) #Bar Plot
plt.show()

sns.boxplot(x='danceability', y='liveness', data=syn_data) #Box plot
plt.show()

plt.scatter(dataset['danceability'],dataset['liveness']) #Scatter Plot
plt.show()

sns.violinplot(x='danceability',y='liveness', data=syn_data) #Violin Plot
#As we can see in the plots, liveness has the highest density when danceabilitiy is between 0.7 and 0.8
#But liveness has its peak when danceability is between 0.2 and 0.3

#Preprocessing:
features=['danceability', 'energy', 'key', 'loudness', 'tempo', 'duration_ms', 'speechiness', 'valence']
target=['liked']
X=np.array(dataset[features].values)
y=np.array(dataset[target].values).ravel()
#Normalization
minX = X.min(axis=0)
maxX = X.max(axis=0)
X = (X - minX) / (maxX - minX)
print("Normalization: ", X[:10])

#Auto selecting
feature_selection = SelectFromModel(LogisticRegression(tol=1e-1))
feature_selection.fit(X, y)

transformedX = feature_selection.transform(X)
print(f"New shape: {transformedX.shape}")
print("Selected features: ", feature_selection.get_support())
print("Selected features: ", np.array(features)[feature_selection.get_support(indices=True)])

feature_selection = SelectFromModel(LinearSVC(tol=1e-1))
feature_selection.fit(X, y)

transformedX = feature_selection.transform(X)
print(f"New shape: {transformedX.shape}")
print("Selected features: ", feature_selection.get_support())
print("Selected features: ", np.array(features)[feature_selection.get_support(indices=True)])

feature_selection = SelectFromModel(DecisionTreeClassifier())
feature_selection.fit(X, y)

transformedX = feature_selection.transform(X)
print(f"New shape: {transformedX.shape}")

print("Selected features: ", feature_selection.get_support())
print("Selected features: ", np.array(features)[feature_selection.get_support(indices=True)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15) # 80% train, %20 test

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=15) # 50% val, 50% test
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in validation dataset: {len(X_valid)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

#Aim is to predict if a song is liked or not

#Model Definition

models = {
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(),
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(kernel='rbf'),
    'kNN': KNeighborsClassifier(n_neighbors=5),
}

for m in models:
  model = models[m]
  model.fit(X_train, y_train)
  score = model.score(X_valid, y_valid)
  print(f'{m} validation score => {score}')

#GaussianNB and kNN has the best validation score

#So I chose kNN:

k_model = KNeighborsClassifier(n_neighbors=9)
k_model.fit(X_train, y_train)

validation_score = k_model.score(X_valid, y_valid)
print(f'Validation score of trained model: {validation_score}')

test_score = k_model.score(X_test, y_test)
print(f'Test score of trained model: {test_score}')


# 9 n_neighbors has more accuracy than 5

#Confusion matrix:
y_predictions = k_model.predict(X_test)
///

conf_matrix = confusion_matrix(y_test, y_predictions)
print(f'Accuracy: {accuracy_score(y_test, y_predictions)}')
print(f'Confussion matrix: \n{conf_matrix}\n')

sns.heatmap(conf_matrix, annot=True)
