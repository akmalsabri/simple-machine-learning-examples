from sklearn import tree, neighbors, svm, naive_bayes
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
clf1 = neighbors.KNeighborsClassifier()
clf2 = svm.SVC()
clf3 = naive_bayes.GaussianNB()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#TEST DATA[height, weight, shoe_size]
test_X = [[179, 90, 44], [190, 88, 44], [165, 55, 37], [160, 60, 39], [156, 56, 36], [181, 85, 43], [174, 66, 40],
     [177, 70, 43], [159, 66, 47], [188, 100, 44], [179, 84, 47]]

test_Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

#CHALLENGE compare their results and print the best one!
prediction = clf.predict(test_X)
prediction1 = clf1.predict(test_X)
prediction2 = clf2.predict(test_X)
prediction3 = clf3.predict(test_X)

accuracy = {'name' : 'DecisionTreeClassifier', 'accuracy' : accuracy_score(test_Y, prediction)}
accuracy1 = {'name' : 'KNeighborsClassifier', 'accuracy' : accuracy_score(test_Y, prediction1)}
accuracy2 = {'name' : 'SVC', 'accuracy' : accuracy_score(test_Y, prediction2)}
accuracy3 = {'name' : 'GaussianNB', 'accuracy' : accuracy_score(test_Y, prediction3)}

if (accuracy['accuracy'] > accuracy1['accuracy'] > accuracy2['accuracy']) and (accuracy['accuracy'] > accuracy1['accuracy'] > accuracy3['accuracy']):
   best = accuracy
elif (accuracy1['accuracy'] > accuracy2['accuracy']) and (accuracy1['accuracy'] > accuracy3['accuracy']):
   best = accuracy1
elif (accuracy2['accuracy'] > accuracy1['accuracy']) and (accuracy2['accuracy'] > accuracy3['accuracy']):
   best = accuracy2
else:
   best = accuracy3

print(best['name'], "works best with the accuracy of", best['accuracy'])