import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
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
    # Load iris data and store in dataframe
    input_file = '0-Datasets/milknew_clear.csv'
    names = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    features = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    target = 'Grade' 
    df = pd.read_csv(input_file, skiprows=3, names=names)
   
    # Separate X and y data
    X = df[features].values
    y = df[target].values

    # Scale the X data using Min-Max scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data - 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Initialize SVM classifier
    svm = SVC(kernel='poly') # poly, rbf, linear
    
    # Training using train dataset
    svm.fit(X_train, y_train)

    # Predict using test dataset
    y_pred_test = svm.predict(X_test)

    # Get accuracy and F1 Score (Holdout)
    accuracy_holdout = accuracy_score(y_test, y_pred_test) * 100
    f1_holdout = f1_score(y_test, y_pred_test, average='macro')
    print("Accuracy (Holdout): {:.4f}%".format(accuracy_holdout))
    print("F1 Score (Holdout): {:.4f}".format(f1_holdout))
    plot_confusion_matrix(y_test, y_pred_test, "Confusion Matrix - Holdout")

    # Perform cross-validation
    y_pred_cv = cross_val_predict(svm, X, y, cv=10)

    # Get accuracy and F1 Score (Cross-Validation)
    accuracy_cv = accuracy_score(y, y_pred_cv) * 100
    f1_cv = f1_score(y, y_pred_cv, average='macro')
    print("Accuracy (Cross-Validation): {:.4f}%".format(accuracy_cv))
    print("F1 Score (Cross-Validation): {:.4f}".format(f1_cv))
    
    plot_confusion_matrix(y, y_pred_cv, "Confusion Matrix - Cross-Validation")

    plt.show()

if __name__ == "__main__":
    main()
