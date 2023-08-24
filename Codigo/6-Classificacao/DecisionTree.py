from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def main():
    input_file = '0-Datasets/milknew_clear.csv'
    names = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    features = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    target = 'Grade' 
    df = pd.read_csv(input_file, skiprows=3, names=names)
   
    # Separating out the features
    X = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:, [target]].values

    # Standardizing the features
    X = MinMaxScaler().fit_transform(X)
    normalizedDf = pd.DataFrame(data=X, columns=features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = DecisionTreeClassifier(max_leaf_nodes=18)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.show()
    
    predictions = clf.predict(X_test)
    print("Holdout Accuracy: {:.2f}%".format(accuracy_score(y_test, predictions) * 100))
    print("Holdout F1 Score:", f1_score(y_test, predictions, average='macro'))
    print("Holdout Confusion Matrix:")
    plot_confusion_matrix(y_test.flatten(), predictions, "Holdout Confusion Matrix")
    
    # Cross-Validation
    clf_cv = DecisionTreeClassifier(max_leaf_nodes=18)
    scores = cross_val_score(clf_cv, X, y.flatten(), cv=10)
    predictions_cv = cross_val_predict(clf_cv, X, y.flatten(), cv=10)
    
    print("Cross-Validation Accuracy: {:.2f}%".format(scores.mean() * 100))
    print("Cross-Validation F1 Score:", f1_score(y.flatten(), predictions_cv, average='macro'))
    print("Cross-Validation Confusion Matrix:")
    plot_confusion_matrix(y.flatten(), predictions_cv, "Cross-Validation Confusion Matrix")

if __name__ == "__main__":
    main()
