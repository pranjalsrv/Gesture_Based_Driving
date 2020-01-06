import xgboost
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC


straight = pickle.load(open("straight_embed.pkl", 'rb'))
right = pickle.load(open("right_embed.pkl", 'rb'))

for i in right:
    print(i)
embeds = np.array(straight + right)
#print(embeds.shape)
targets = np.array([0]*len(straight) + [1]*len(right))


x_train,x_test, y_train,y_test = train_test_split(embeds, targets, test_size=0.3, shuffle=True)
# model = xgboost.XGBClassifier()
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# acc = accuracy_score(y_test,y_pred)
# print("Accuracy = ",acc)

model2 = SVC(kernel='linear')
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("Accuracy = ",acc)
