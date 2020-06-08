import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression

path_to_videos = "InputDataCSV\\"

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

init = True
main_data = None

length = []
files = os.listdir(path_to_videos)
t = 80
for file in files:
    if not os.path.isdir(path_to_videos + file + "/"):
        new_path = path_to_videos + file
        df = pd.read_csv(new_path)
        data = df.values
        data = data[:,1:]
        rows = data.shape[0]
        if rows < t:
            temp = np.zeros((t-data.shape[0], data.shape[1]))
            data = np.concatenate((data,temp))
        else:
            data = data[:t,:]

        length.append(data.shape[0])
        row = data.shape[0]
        col = data.shape[1]

        data = data.reshape(1, row*col)

        if init:
            main_data = data
            init = False
        else:
            main_data = np.concatenate((main_data, data))


def classifier(X):

    y = [0]*71
    y.extend([1]*71)
    y.extend([2] * 70)
    y.extend([3] * 68)
    y.extend([4] * 65)
    y.extend([5] * 70)
    y = np.array(y)

    models = [LogisticRegression(solver='lbfgs', max_iter=2000), SVC(kernel="linear", C=0.025),
              LogisticRegression(solver='lbfgs', max_iter=2000), SVC(kernel="linear", C=0.025)]

    names = ["Logistic Regression", "SVC","Logistic Regression", "SVC"]

    kf = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)

    for i in range(len(models)):
        best_accuracy, best_model = 0, 0
        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = models[i]
            clf.fit(X_train, y_train)
            svm_predictions = clf.predict(X_test)
            print(svm_predictions)

            # model accuracy for X_test
            accuracy = clf.score(X_test, y_test)*100
            print(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = clf

        # Save the weights to a pickle file
        filename = "best_model_"+str(i)+".sav"
        pickle.dump(best_model, open(filename, 'wb'))
        print("Best accuracy for ", names[i], " is ", best_accuracy)


classifier(main_data)

