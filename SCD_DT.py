from sklearn import tree
import pydotplus
from sklearn.datasets import load_iris
from IPython.display import Image

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf1 = tree.DecisionTreeClassifier()
clf1= clf1.fit(X, Y)
with open("graph.dot", 'w') as f:
    f = tree.export_graphviz(clf1, out_file=f)
clf1.predict([[2., 2.]])
clf1.predict_proba([[2., 2.]])

iris = load_iris()
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(iris.data, iris.target)

# export the tree in Graphviz format using the export_graphviz exporter
with open("iris.dot", 'w') as f:
    dot_data = tree.export_graphviz(clf2, out_file=f,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)

# predict the class of samples
clf2.predict(iris.data[:1, :])
# the probability of each class
clf2.predict_proba(iris.data[:1, :])

