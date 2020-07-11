import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#Import Data
df = pd.read_csv('wineclean.csv')
#print(df.columns)

#Create smaller subsets for quality:
rating = []
for i in df['quality']:
    if i >= 1 and i < 6:
        rating.append('0')
    elif i >= 6 and i < 7:
        rating.append('1')
    elif i >= 7:
        rating.append('2')
df['rating'] = rating
#These ratings roughly will correspond with 0-Bad, 1-Good, 2-Exceptional

#Create dummy variables
df_model = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'rating']]


#Create train and test split:
X= df_model.drop('rating', axis=1)
y= df_model.rating.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#print(X_train.shape)

#Import Packages for Models
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


#Create Logistic Regression and Train:
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)

#Evaluate the X test set for Logisitic Regression:
lr_matrix = confusion_matrix(y_test, lr_predictions)
lr_acc = accuracy_score(y_test, lr_predictions)
print('Logistic Regression Accuracy:')
print(lr_acc*100)

plt.figure()
ax= plt.subplot()
ax = sns.heatmap(lr_matrix, annot=True, fmt='g', ax = ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix for Logistic Regression');
ax.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig = ax.get_figure()
fig.savefig('LogisticRegressionConfusionMatrix.png')

#Create Gradient Boosting Classifier and Train:
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)
gbc_predict = gbc.predict(X_test)

#Evaluate the X test set for Gradient Boosting:
gbc_matrix = confusion_matrix(y_test, gbc_predict)
gbc_acc = accuracy_score(y_test, gbc_predict)
print('Gradient Boosting Classifier Accuracy:')
print(gbc_acc*100)

ax.remove()
ax= plt.subplot()
ax2 = sns.heatmap(gbc_matrix, annot=True, fmt='g', ax = ax);
ax2.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax2.set_title('Confusion Matrix for Gradient Boosting');
ax2.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig2 = ax2.get_figure()
fig2.savefig('GradientBoostingClassifierConfusionMatrix.png')

#Create Decision Tree Classifier and Train:
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_predict = dt.predict(X_test)

#Evaluate the X test set for Decision Tree Classifier:
dt_matrix = confusion_matrix(y_test, dt_predict)
dt_acc = accuracy_score(y_test, dt_predict)
print('Decision Tree Accuracy:')
print(dt_acc*100)

ax.remove()
ax= plt.subplot()
ax3 = sns.heatmap(dt_matrix, annot=True, fmt='g', ax = ax);
ax3.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax3.set_title('Confusion Matrix for Decision Tree Classifier');
ax3.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig3 = ax3.get_figure()
fig3.savefig('DecisionTreeClassifierConfusionMatrix.png')

#Create AdaBoost and Train:
ada = AdaBoostClassifier(random_state=42)
ada.fit(X_train, y_train)
ada_predict = ada.predict(X_test)

#Evaluate the X test set for AdaBoost Classifier:
ada_matrix = confusion_matrix(y_test, ada_predict)
ada_acc = accuracy_score(y_test, ada_predict)
print('AdaBoost Accuracy:')
print(ada_acc*100)

ax.remove()
ax= plt.subplot()
ax4 = sns.heatmap(ada_matrix, annot=True, fmt='g', ax = ax);
ax4.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax4.set_title('Confusion Matrix for AdaBoost');
ax4.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig4 = ax4.get_figure()
fig4.savefig('AdaBoostConfusionMatrix.png')

#Create NaiveBayes and Train:
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predict = nb.predict(X_test)

#Evaluate the X test set for NaiveBayes:
nb_matrix = confusion_matrix(y_test, nb_predict)
nb_acc = accuracy_score(y_test, nb_predict)
print('NaiveBayes Accuracy:')
print(nb_acc*100)

ax.remove()
ax= plt.subplot()
ax5 = sns.heatmap(nb_matrix, annot=True, fmt='g', ax = ax);
ax5.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax5.set_title('Confusion Matrix for NaiveBayes');
ax5.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig5 = ax5.get_figure()
fig5.savefig('NaiveBayesConfusionMatrix.png')

#Create Random Forest and Train:
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)

#Evaluate Random Forest:
#Evaluate the X test set for Decision Tree Classifier:
rf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc = accuracy_score(y_test, rf_predict)
print('Random Forest Accuracy:')
print(rf_acc*100)

ax.remove()
ax= plt.subplot()
ax6 = sns.heatmap(rf_matrix, annot=True, fmt='g', ax = ax);
ax6.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax6.set_title('Confusion Matrix for Decision Tree Classifier');
ax6.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig6 = ax6.get_figure()
fig6.savefig('RandomForestConfusionMatrix.png')

#Create SVC and Train:
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_predict = svc.predict(X_test)

#Evaluate the X test set for SVC:
svc_matrix = confusion_matrix(y_test, svc_predict)
svc_acc = accuracy_score(y_test, svc_predict)
print('SVC Accuracy:')
print(svc_acc*100)

ax.remove()
ax= plt.subplot()
ax7 = sns.heatmap(svc_matrix, annot=True, fmt='g', ax = ax);
ax7.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax7.set_title('Confusion Matrix for SVC');
ax7.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig7 = ax7.get_figure()
fig7.savefig('SVCConfusionMatrix.png')

#Create Linear SVC and Train:
svcl = LinearSVC(random_state=42)
svcl.fit(X_train, y_train)
svcl_predict = svcl.predict(X_test)

#Evaluate the X test set for Linear SVC:
svcl_matrix = confusion_matrix(y_test, svcl_predict)
svcl_acc = accuracy_score(y_test, svcl_predict)
print('Linear SVC Accuracy:')
print(svcl_acc*100)

ax.remove()
ax= plt.subplot()
ax8 = sns.heatmap(svcl_matrix, annot=True, fmt='g', ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax8.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax8.set_title('Confusion Matrix for Linear SVC');
ax8.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig8 = ax8.get_figure()
fig8.savefig('LinearSVCConfusionMatrix.png')

#Create K Neighbors and Train:
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
kn_predict = kn.predict(X_test)

#Evaluate the X test set for K Neighbors:
kn_matrix = confusion_matrix(y_test, kn_predict)
kn_acc = accuracy_score(y_test, kn_predict)
print('K Neighbors Accuracy:')
print(kn_acc*100)

ax.remove()
ax= plt.subplot()
ax9 = sns.heatmap(kn_matrix, annot=True, fmt='g', ax = ax);
ax9.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax9.set_title('Confusion Matrix for K Neighbors');
ax9.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig9 = ax9.get_figure()
fig9.savefig('KNeighborsConfusionMatrix.png')

''' Top 3 Results are:
Random Forest Accuracy: 69.79
Gradient Boosting Classifier: 65.83
Logistic Regression Classifier: 58.13
Lets run optimization functions for RF.
'''
from sklearn.model_selection import GridSearchCV

print('Grid Searching Random Forest Model')
RFparams = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200, 3600, 4000]}
#GSRF = GridSearchCV(rf, RFparams, cv=3, n_jobs=-1)
#GSRF.fit(X_train, y_train)
#RFmaxscore = GSRF.best_score_
'''RFBestEst = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=20, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=2000,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
'''

RFBestEst = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=20, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=1000,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)

#print(RFmaxscore)
print(RFBestEst)

RFBestEst.fit(X_train, y_train)
y_test = y_test.reshape(-1,1)
RFBestPredict= RFBestEst.predict(X_test)

rfbest_matrix = confusion_matrix(y_test, RFBestPredict)
rfbest_acc = accuracy_score(y_test, RFBestPredict)
print('Best Random Forest Accuracy:')
print(rfbest_acc*100)

ax.remove()
ax= plt.subplot()
ax10 = sns.heatmap(rfbest_matrix, annot=True, fmt='g', ax = ax);
ax10.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax10.set_title('Confusion Matrix for Best Random Forest');
ax10.xaxis.set_ticklabels(['Bad', 'Good','Exceptional']); ax.yaxis.set_ticklabels(['Bad', 'Good','Exceptional']);
fig10 = ax10.get_figure()
fig10.savefig('BestForestConfusionMatrix.png')


from sklearn.externals import joblib
filename = 'RandomForestOptimized.pkl'
joblib.dump(RFBestEst, filename)