import xgboost
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.svm import SVC


straight = pickle.load(open("straight_embed.pkl", 'rb'))
right = pickle.load(open("right_embed.pkl", 'rb'))

embeds = np.array(straight + right)
#print(embeds.shape)
targets = np.array([0]*len(straight) + [1]*len(right))

print("Embedding shape = ", embeds.shape)
nsamples, nx,ny, each = embeds.shape
train_2 = embeds.reshape((nsamples,nx*ny*each))
print("Flattened embeddings =",train_2.shape)


x_train,x_test, y_train,y_test = train_test_split(train_2, targets, test_size=0.3, shuffle=True)
model = xgboost.XGBClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("Accuracy = ",acc)
clf_report = classification_report(y_test,y_pred)
print(clf_report)

# model2 = SVC(kernel='linear')
# model2.fit(x_train, y_train)
# y_pred = model2.predict(x_test)
# acc = accuracy_score(y_test,y_pred)
# print("Accuracy = ",acc)
