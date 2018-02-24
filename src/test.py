from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc, roc_curve
from elm import ELMClassifier,GenELMClassifier
from random_layer import RandomLayer,MLPRandomLayer,RBFRandomLayer,GRBFRandomLayer

import numpy as np
import pandas as pd
import seaborn

# 335 OTUs in total and 490 samples
otuinfile = 'glne007.final.an.unique_list.0.03.subsample.0.03.filter.shared'
mapfile = 'metadata.tsv'
disease_col = 'dx'

# Data reading
data = pd.read_table(otuinfile,sep='\t',index_col=1)
filtered_data = data.dropna(axis='columns', how='all')
X = filtered_data.drop(['label','numOtus'],axis=1)
metadata = pd.read_table(mapfile,sep='\t',index_col=0)
y = metadata[disease_col]
## Merge adenoma and normal in one-category called no-cancer, so we have binary classification
y = y.replace(to_replace=['normal','adenoma'], value=['no-cancer','no-cancer'])

encoder = LabelEncoder()
y = pd.Series(encoder.fit_transform(y),
index=y.index, name=y.name)

A, P, Y, Q = train_test_split(
X, y, test_size=0.15, random_state=42)	# Can change to 0.2


srhl_rbf = RBFRandomLayer(n_hidden=50,rbf_width=0.1,random_state=0)
clf6 = GenELMClassifier(hidden_layer=srhl_rbf).fit(A, Y.values.ravel())
print ("Accuracy of Extreme learning machine Classifier: "+str(clf6.score(P,Q)))


#==============================================
plt.figure()
cls = 0
# Set figure size and plot layout
figsize=(20,15)
f, ax = plt.subplots(1, 1, figsize=figsize)

x = [clf6,'purple','ELM']

#y_true = Q[Q.argsort().index]
y_score = x[0].decision_function(P)
#y_prob = x[0].predict_proba(P.ix[Q.argsort().index, :])
fpr, tpr, _ = roc_curve(Q, y_score)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color=x[1], alpha=0.8,
        label='Test data: {} '
              '(auc = {:.2f})'.format(x[2], roc_auc))

ax.set_xlabel('False Positive Rate',fontsize=15)
ax.set_ylabel('True Positive Rate',fontsize=15)
ax.legend(loc="lower right",fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
plt.show()