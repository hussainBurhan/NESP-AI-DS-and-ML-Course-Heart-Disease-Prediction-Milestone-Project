# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics

# Importing SKLearn Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Importing Evaluation Libraries
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, recall_score, f1_score, RocCurveDisplay

# Read the dataset
df = pd.read_csv('heart.csv')

# Display basic information about the dataset
print(f'Shape : {df.shape}')
print('Head:')
print(df.head())
print('Tail:')
print(df.tail())

# Plot the distribution of target variable
df['target'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.show()

# Display information about missing values and basic statistics of the dataset
print('Missing Values:')
print(df.isna().sum())
print('Description of Data:')
print(df.describe())

# Display a cross-tabulation of target and sex, and plot the results
print(pd.crosstab(df.target, df.sex))
pd.crosstab(df.target, df.sex).plot(kind='bar', figsize=(10, 6), color=['red', 'green'])
plt.title('Heart Disease summary sex wise')
plt.xlabel('0 = No Heart Disease, 1 = Heart Disease')
plt.ylabel('Number of Individuals')
plt.legend(['Female', 'Male'])
plt.xticks(rotation=0)
plt.show()

# Create a scatter plot for Age vs Heart Rate, and a histogram for Age
plt.figure(figsize=(10, 6))
plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c='red')
plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c='green')
plt.title('Age Vs Heart Rate')
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.legend(['Heart Problem', 'No Heart Problem'])
plt.show()
df.age.plot.hist()
plt.show()

# Display a cross-tabulation of chest pain and target, and plot the results
print(pd.crosstab(df.cp, df.target))
pd.crosstab(df.cp, df.target).plot(kind='bar', figsize=(10, 6), color=['green', 'red'])
plt.title('Chest Pain Vs Heart Disease')
plt.xlabel('Chest Pain Type')
plt.ylabel('Counts')
plt.legend(['No Hear Problem', 'Heart Problem'])
plt.show()

# Create a heatmap of the correlation matrix
corr_map = df.corr()
print(corr_map)
home, room = plt.subplots(figsize=(10, 10))
room = sns.heatmap(corr_map,
                   annot=True,
                   linewidths=0.5,
                   fmt='0.2f',
                   cmap='YlGnBu')
plt.show()

# Split the data into features (x) and target variable (y), and perform train-test split
x = df.drop('target', axis=1)
y = df['target']
np.random.seed(7)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define a dictionary of models for comparison
models = {'KNN': KNeighborsClassifier(),
          'Random Forest': RandomForestClassifier(),
          'Logistic Regression': LogisticRegression()}

# Define a function to fit and score the models
def fit_and_score(models, x_train, x_test, y_train, y_test):
    np.random.seed(7)
    model_score = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        model_score[name] = model.score(x_test, y_test)
    return model_score

# Evaluate and compare the models
model_scores = fit_and_score(models=models, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print(model_scores)
model_compare = pd.DataFrame(model_scores, index=['Accuracy'])
model_compare.plot.bar()
plt.show()

# Hyperparameter tuning for KNN
train_score = []
test_score = []
nneighbors = range(1, 30)
KNN = KNeighborsClassifier()
for i in nneighbors:
    KNN.set_params(n_neighbors=i)
    KNN.fit(x_train, y_train)
    train_score.append(KNN.score(x_train, y_train))
    test_score.append(KNN.score(x_test, y_test))

print(f'Training Scores: {train_score}')
print(f'Test Score: {test_score}')
plt.plot(nneighbors, train_score, label='Train Score')
plt.plot(nneighbors, test_score, label='Test Score')
plt.xticks(np.arange(1, 31, 1))
plt.xlabel('N-Neighbors')
plt.ylabel('Model Accuracy')
plt.legend()
print(f'Max KNN Score: {max(test_score)}')
plt.show()

# Hyperparameter tuning using RandomizedSearchCV for Logistic Regression and Random Forest
LR_hp = {'C': np.logspace(-4, 4, 20),
         'solver': ['liblinear']}
RFC_hp = {'n_estimators': np.arange(100, 1000, 100),
          'max_depth': [None, 5, 6, 10],
          'min_samples_split': np.arange(2, 20, 2),
          'min_samples_leaf': np.arange(2, 10, 2)}

np.random.seed(7)
lr = RandomizedSearchCV(LogisticRegression(),
                        param_distributions=LR_hp,
                        cv=5,
                        n_iter=20,
                        verbose=True)
lr.fit(x_train, y_train)
print(f'Logistic Regression RSCV Score: {lr.score(x_test, y_test)}')
print(f'Logistic Regression Best Parameters: {lr.best_params_}')

np.random.seed(7)
rfc = RandomizedSearchCV(RandomForestClassifier(),
                         param_distributions=RFC_hp,
                         cv=5,
                         n_iter=20,
                         verbose=True)
rfc.fit(x_train, y_train)
print(f'Random Forest Classifier RSCV Score: {rfc.score(x_test, y_test)}')
print(f'Random Forest Classifier Best Parameters: {rfc.best_params_}')

# Hyperparameter tuning using GridSearchCV for Logistic Regression and Random Forest
np.random.seed(7)
lr_gs = GridSearchCV(LogisticRegression(),
                     param_grid=LR_hp,
                     cv=5,
                     verbose=True)
lr_gs.fit(x_train, y_train)
print(f'Logistic Regression GSCV Score: {lr_gs.score(x_test, y_test)}')
print(f'Logistic Regression Best Parameters: {lr_gs.best_params_}')

np.random.seed(7)
rfc_gs = GridSearchCV(RandomForestClassifier(),
                      param_grid=RFC_hp,
                      cv=5,
                      verbose=True)
rfc_gs.fit(x_train, y_train)
print(f'Random Forest Classifier GSCV Score: {rfc_gs.score(x_test, y_test)}')
print(f'Random Forest Classifier Best Parameters: {rfc_gs.best_params_}')

# Plot the ROC curve for Logistic Regression
RocCurveDisplay.from_estimator(lr_gs, x_test, y_test)
plt.show()

# Plot the confusion matrix for Logistic Regression
y_pred = lr_gs.predict(x_test)

# Define a function to plot the confusion matrix
def plot_confusion(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=False)

plot_confusion(y_test, y_pred)
plt.show()

# Display classification report for Logistic Regression
print(classification_report(y_test, y_pred))

# Cross-validation for Logistic Regression and display metrics
lr = LogisticRegression(C=0.615848211066026, solver='liblinear')
cv_accuracy = cross_val_score(lr, x, y, cv=7, scoring='accuracy')
cv_accuracy = np.mean(cv_accuracy)
print(f'Accuracy: {cv_accuracy}')

cv_precision = cross_val_score(lr, x, y, cv=7, scoring='precision')
cv_precision = np.mean(cv_precision)
print(f'Precision: {cv_precision}')

cv_recall = cross_val_score(lr, x, y, cv=7, scoring='recall')
cv_recall = np.mean(cv_recall)
print(f'Recall: {cv_recall}')

cv_f1 = cross_val_score(lr, x, y, cv=7, scoring='f1')
cv_f1 = np.mean(cv_f1)
print(f'F1: {cv_f1}')

# Display a comparison bar plot of metrics
cv_plot = pd.DataFrame({'Accuracy': cv_accuracy,
                        'Precision': cv_precision,
                        'Recall': cv_recall,
                        'F1': cv_f1}, index=[0])
cv_plot.T.plot.bar(title='Matrix Comparison', legend=False)
plt.show()

# Fit Logistic Regression on the training data and plot feature importance
lr.fit(x_train, y_train)
final_coef = dict(zip(df.columns, list(lr.coef_[0])))
features = pd.DataFrame(final_coef, index=[0])
features.T.plot.bar(title='Features', legend=False)
plt.show()
