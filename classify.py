from sklearn import tree

feature = [[120,5] , [110,5] , [180,12] , [163,12] ,[93,5]]
lables = ["ECO CAR / B-SEGMENT" , "ECO CAR / B-SEGMENT" , "VAN" , "VAN-SEGMENT" , "Van-BSEGMENT"]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(feature , lables)

print(classifier.predict([[90 , 5]]))


