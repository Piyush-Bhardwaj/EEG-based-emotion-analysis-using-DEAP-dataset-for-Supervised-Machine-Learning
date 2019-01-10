import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

 
def svm_classifier_pca():
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_0.dat'
    print("LABEL 0 - Valence \n ")
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    print("Split the data into training/testing sets \n")
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test_0 = train_test_split(X, y, test_size=0.33, random_state=42)
     
    
    print("Feature Scaling \n")
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    print("Applying PCA to select features \n")
    
    # PCA to select features
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    #explained_variance=pca.explained_variance_ratio_
    
    print("Applying SVM classifier \n")
    # SVM Classifier
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict_0 = clf.predict(X_test)
    cm = confusion_matrix(y_test_0, y_predict_0)
    print(cm)
    print("Accuracy score of Valence SVM-PCA")
    print(accuracy_score(y_test_0, y_predict_0)*100)
    
    #######################################################################
    
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_1.dat'
    print("LABEL 1 - Arousal \n ")
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    print("Split the data into training/testing sets \n")
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test_0 = train_test_split(X, y, test_size=0.33, random_state=42)
     
    
    print("Feature Scaling \n")
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    print("Applying PCA to select features \n")
    
    # PCA to select features
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    #explained_variance=pca.explained_variance_ratio_
    
    print("Applying SVM classifier \n")
    # SVM Classifier
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict_0 = clf.predict(X_test)
    cm = confusion_matrix(y_test_0, y_predict_0)
    print(cm)
    print("Accuracy score of Arousal SVM-PCA")
    print(accuracy_score(y_test_0, y_predict_0)*100)
    
    #######################################################################
    
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_2.dat'
    print("LABEL 2 - Dominance \n ")
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    print("Split the data into training/testing sets \n")
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test_0 = train_test_split(X, y, test_size=0.33, random_state=42)
     
    
    print("Feature Scaling \n")
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    print("Applying PCA to select features \n")
    
    # PCA to select features
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    #explained_variance=pca.explained_variance_ratio_
    
    print("Applying SVM classifier \n")
    # SVM Classifier
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict_0 = clf.predict(X_test)
    cm = confusion_matrix(y_test_0, y_predict_0)
    print(cm)
    print("Accuracy score of Dominance SVM-PCA")
    print(accuracy_score(y_test_0, y_predict_0)*100)
    
    #######################################################################
    
    
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_0.dat'
    print("LABEL 3 - Liking \n ")
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    print("Split the data into training/testing sets \n")
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test_0 = train_test_split(X, y, test_size=0.33, random_state=42)
     
    
    print("Feature Scaling \n")
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    print("Applying PCA to select features \n")
    
    # PCA to select features
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    #explained_variance=pca.explained_variance_ratio_
    
    print("Applying SVM classifier \n")
    # SVM Classifier
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict_0 = clf.predict(X_test)
    cm = confusion_matrix(y_test_0, y_predict_0)
    print(cm)
    print("Accuracy score of Liking SVM-PCA")
    print(accuracy_score(y_test_0, y_predict_0)*100)
    
   
if __name__ == '__main__':
    svm_classifier_pca()