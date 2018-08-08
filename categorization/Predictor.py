from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def level_predict(X_train, X_test, y_train):
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    return prediction


def evaluate(prediction, y_test):
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
