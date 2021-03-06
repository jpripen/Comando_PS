import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp

###################
# DATA PRESENTATION
###################
print('data presentation'.upper())
print('-' * 20)

# Importing files
df_train = pd.read_csv('Titanic/data/train.csv')
df_test = pd.read_csv('Titanic/data/test.csv')

print(f'DATASET SIZES\nTrain set: {df_train.shape}\nTest set: {df_test.shape}\n')

# To study the full set, let's merge both datasets and add to the test set a column named 'Survived', 
# with NaN values in it
survive = np.empty((df_test.shape[0] ,1))
survive[:] = np.nan
df_test.insert(1, 'Survived', survive)

# Appending data frames and setting passenger ID as index
df_full = df_train.append(df_test, ignore_index = True)
df_full.reset_index()
df_full.set_index('PassengerId', inplace = True) 

# Print a random sample of the full dataset
print('Random sample of the full dataset:\n'.upper())
print(df_full.sample(10))
print('\n')

#####################
# FEATURE ENGINEERING
#####################
print('feature engineering'.upper())
print('-' * 20)

# Counting missing values
print('Counting missing values:\n'.upper())
print(f'The full dataset has {df_full.shape[0]} entries\n')
print(df_full.isnull().sum())
print('\n')

# TITLE INFORMATION
import re

def get_title(name):
	"""If the title exists, extract and return it."""

	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""

# We create a new column with each passengers Title, if any.
df_full['Title'] = df_full['Name'].apply(get_title)

print('unique titles before reassignment\n'.upper())
print(df_full.Title.unique())
print('\n')

# Making groups of Titles by hand
df_full['Title'] = df_full['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Major', 'Rev', 'Sir', 'Jonkheer'], 
											'Noble')   
df_full['Title'] = df_full['Title'].replace('Don', 'Mr')
df_full['Title'] = df_full['Title'].replace(['Mlle', 'Ms'], 'Miss')
df_full['Title'] = df_full['Title'].replace(['Mme', 'Dona'], 'Mrs')

male_dr_filter = (df_full.Title == 'Dr') & (df_full.Sex == 'male')
female_dr_filter = (df_full.Title == 'Dr') & (df_full.Sex == 'female')
df_full.loc[male_dr_filter, ['Title']] = 'Mr'
df_full.loc[female_dr_filter, ['Title']] = 'Mrs'

# Checking results
print('unique titles after reassignment, grouped by gender\n'.upper())
print(pd.crosstab(df_full['Title'], df_full['Sex']))
print('\n')

# AGE
# Before filling age missing values we want to segment age groups by using a KMeans clustering algorithm
from sklearn.cluster import KMeans

age_data = df_full[['Age', 'Survived']].dropna()

# First of all, we'll use the Elbow Method to find the optimal number of clusters
# THIS CODE WAS USED TO GENERATE ElbowPlot.png
#sum_squared_distances = []
#k_range = range(1, 11)
#for k in k_range:
#    km = KMeans(n_clusters = k)
#    km = km.fit(age_data)
#	# Computing squared distances of samples to their closest cluster center
#    sum_squared_distances.append(km.inertia_)

#plt.plot(k_range, sum_squared_distances, 'bx-')
#plt.xlabel('Number of clusters (k)')
#plt.ylabel('Sum of squared distances')
#plt.title('Elbow Method for optimal number of clusters')
#plt.savefig('ElbowPlot.png', dpi = 100)
#plt.show()

# We can perform a KMeans with k = 4 as suggested by the Elbow Method
k_elbow = 4
random_state = 4
kmeans = KMeans(n_clusters = k_elbow, random_state = random_state)
kmeans.fit(age_data)
age_data['Label'] = kmeans.labels_

# And obtain the age bands to create a new feature AgeBand
age_bands = age_data.groupby('Label')['Age'].min().astype(int).sort_values().tolist()
age_bands.append(np.inf) # Adding an upper boundary

print('Age boundaries obtained after KMeans clustering:'.upper())
print(age_bands)
print('\n')

# We can fill the missing values now. To fill the data, we will calculate the median as a function 
# of Title and Pclass using only the training data [:891]
ages_table = df_full[:891].groupby(['Title', 'Pclass'])['Age'].median()

print('median age by title and class'.upper())
print(ages_table)
print('\n')

# Replacing NaN with zeros and then filling with the median for its Title/Pclass
df_full['Age'] = df_full['Age'].fillna(0)

for ind, row in df_full[df_full['Age'] == 0].iterrows():
    df_full.loc[ind, 'Age'] = ages_table[row.Title][row.Pclass]

# AGE BAND
df_full['AgeBand'] = pd.cut(df_full['Age'], age_bands)

print('survival rate for each age group'.upper())
print(df_full.groupby('AgeBand')['Survived'].mean())
print('\n')

# FAMILIY SIZE
df_full['FamilySize'] = df_full['SibSp'] + df_full['Parch'] + 1

# TICKET OCCURRENCE
df_full['TicketOccurr'] = df_full.groupby('Ticket')['Ticket'].transform('size')

# IS ALONE
# We'll define it from Ticket Occurrence instead of Family Size
df_full['IsAlone'] = df_full['TicketOccurr'].apply(lambda x: 1 if x == 1 else 0)

# FARE PER PERSON
df_full['FarePerPerson'] = df_full['Fare'] / df_full['TicketOccurr']

# As we did with missing ages, we'll fill the FarePerPerson missing values by using the median of the Pclass
fares_table = df_full[:891].groupby('Pclass')['FarePerPerson'].median()

print('median fare by class'.upper())
print(fares_table)
print('\n')

for ind, row in df_full[df_full['FarePerPerson'].isnull()].iterrows():
	df_full.loc[ind, 'FarePerPerson'] = fares_table[row.Pclass]

# EMBARKED
# We'll fill the Embarked missing values we can use the MODE, we can take a random sample or we can study if there is
# any correlation between Embarked and other features such as FarePerPerson, Title or Pclass

# What is the proportion of passengers of each class that have embarked at each port?
df_ports = df_full[:891].groupby(['Embarked', 'Pclass'])['Pclass'].count()

print('number of passengers for each port/class'.upper())
print(df_ports)
print('\n')

df_ports_prop = df_ports / df_full[:891].groupby(['Embarked'])['Pclass'].count()

print('proportion of passengers for each port/class'.upper())
print(df_ports_prop)
print('\n')

# We decided (for now) to fill the missing values by randomly sampling the Embarked column 
num_nan = df_full[df_full[['Embarked']].isnull().any(axis=1)].index
port_samp = list(df_train['Embarked'].sample(len(num_nan), replace = True))

for i in range(len(num_nan)):
    df_full.loc[num_nan[i], 'Embarked'] = port_samp[i]

# Checking missing values again
print('Counting missing values after imputation:\n'.upper())
print(f'The full dataset has {df_full.shape[0]} entries\n')
print(df_full.isnull().sum())
print('\n')

## AGE * PCLASS
## We'll add a new feature based on Age and Pclass 
#df_full['Age*Pclass'] = df_full['Age'] * df_full['Pclass']

#print(df_full['Age*Pclass'].unique())

# Finally, we transform categorical variables to dummies
columns = ['AgeBand', 'Pclass', 'Title', 'Embarked']
df_full = pd.get_dummies(df_full, columns = columns)

df_full['Sex'] = df_full['Sex'].map({'male':1,'female':0})

# Generate the correlation matrix for the train set
#corr_full = df_full.corr()

#corr_fig = plt.figure(figsize = (10, 10))
#sns.heatmap(corr_full, annot = True)
#plt.title("Titanic survivor correlation matrix heatmap")
#plt.show()

# Exporting df_full for modelling
# df_full.to_csv('titanic_features.csv')

# THIS CODE WAS USED TO GENERATE AgeBands.png
# Plot the decission boundaries
# See http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#plt.figure(figsize=(10, 5))
#h = 0.01
#x_min, x_max = age_data['Age'].min() - h, age_data['Age'].max() + h
#y_min, y_max = age_data['Survived'].min() - h, age_data['Survived'].max() + h
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

## Predict the age cluster for each point in a mesh
#Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
## Put the result into a color plot
#cmap = sns.cubehelix_palette(start=2.8, rot=.1, as_cmap=True)
#Z = Z.reshape(xx.shape)
#plt.imshow(Z, interpolation='nearest',
#           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#           cmap=cmap, aspect='auto')
## Plot the ages
#sns.scatterplot(x='Age', y='Survived', hue='Label', data=age_data, palette=cmap)
## Plot the centroids as a white X
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='w')
#plt.yticks([0, 1])
#plt.title("Age clusters and decision boundaries")
#plt.savefig('AgeBands.png', dpi = 100)
#plt.show()

##########
# MODELING
##########

# First of all, we select relevant features
data = df_full.drop(['Survived', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'FamilySize', 
					 'TicketOccurr', 'FarePerPerson'], axis = 1)

print(data.columns)
print(data.head())

# Now, we can split the data
train = data[0:891]        # "original" train set containing transformed/selected features
test = data[891:]          # "original" test set containing transformed/selected features 
target = df_train.Survived # being df_train the original train data we imported in the beginning 

print(f'''Checking data frame sizes...\ntrain has {train.shape} entries\ntest has {test.shape} entries 
target has { target.shape} entries\n''')

# We should now preprocess our data before using any algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = StandardScaler().fit_transform(train)
y = target

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

print(f'''Checking train and test split sizes...
Train set: X_train -> {X_train.shape}, y_train -> {y_train.shape}
Test set: X_test -> {X_test.shape}, y_test -> {y_test.shape}\n''')

# Model selection
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.model_selection import cross_val_score

# Set random state and number of estimators for tree based models
random_state = 4
n_estimators = 100

models = [LogisticRegression(
             random_state = random_state),
          Perceptron(
              random_state = random_state), 
          SGDClassifier(
              random_state = random_state), 
          SVC(
              random_state = random_state), 
          KNeighborsClassifier(
              ), 
          GaussianNB(
              ),
          DecisionTreeClassifier(
              random_state = random_state), 
          RandomForestClassifier(
              random_state = random_state,
              n_estimators = n_estimators),
          ExtraTreesClassifier(
              random_state = random_state,
              n_estimators = n_estimators),
          AdaBoostClassifier(
              random_state = random_state,
              n_estimators = n_estimators),
          GradientBoostingClassifier(
              random_state = random_state, 
              n_estimators = n_estimators)
         ]

# Lists to store the results
model_name = []
acc_test = []
acc_train = []
cv_scores = []

# Number of folds for the cross validation
cv = 5

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_name.append(model.__class__.__name__)
    acc_test.append(model.score(X_test, y_test))
    acc_train.append(model.score(X_train, y_train))
    # We use X and y here because cross_val_score needs all the data to perform the splits by itself
    cv_scores.append(cross_val_score(model, X, y, cv = cv))

results = pd.DataFrame({
    'Model': model_name,
    'CVScore' : cv_scores,
    'TestScore': acc_test,
    'TrainScore': acc_train
    })

# We need to obtain the mean value for the CVScore column
results.insert(1, 'CVMeanScore', np.mean(results['CVScore'].tolist(), axis = 1))
results.drop(['CVScore'], axis = 1, inplace = True)

print('Cross validation results'.upper())
print(results)

# Finally, we select the top 5 algorithms based on the CVMeanScore
results_top5 = results.sort_values(by = 'CVMeanScore', ascending = False, ignore_index = True).head()

print('\nTop 5 algorithms'.upper())
print(results_top5)

# Making predictions for Stacking
# We have to split the original X and y matrices again: first and second refer to the base model and final outcome layer
X_first, X_second, y_first, y_second = train_test_split(X, y, test_size = 0.5, random_state = 4)

best_models = [GradientBoostingClassifier(
              random_state = random_state, 
              n_estimators = n_estimators),
	          SVC(
              random_state = random_state),
              KNeighborsClassifier(
              ),
              DecisionTreeClassifier(
              random_state = random_state),
              ExtraTreesClassifier(
              random_state = random_state,
              n_estimators = n_estimators)]

predictions = pd.DataFrame()

for model in best_models:
    model.fit(X, y)
    predictions.insert(0, model.__class__.__name__, model.predict(X))

predictions.insert(len(predictions.columns), 'Survived', y)

print(predictions.sample(10))

# We perform the second layer fitting
import xgboost as xgb
from sklearn.metrics import accuracy_score

features = predictions.drop('Survived', axis = 1)
outcome = predictions.Survived

f_train, f_test, o_train, o_test = train_test_split(features, outcome, test_size = 0.2, random_state = random_state)


xgb_model = xgb.XGBClassifier(random_state = random_state)
xgb_model.fit(f_train, o_train)
o_pred = xgb_model.predict(f_test)

print(accuracy_score(o_test, o_pred))

# Submission csv
X_stack = StandardScaler().fit_transform(test)

stack_preds = pd.DataFrame()

for model in best_models:
    stack_preds.insert(0, model.__class__.__name__, model.predict(X_stack))

final_out = xgb_model.predict(stack_preds)

submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': final_out})
print(submission)

submission.to_csv("StackingSubmission.csv", index = False)