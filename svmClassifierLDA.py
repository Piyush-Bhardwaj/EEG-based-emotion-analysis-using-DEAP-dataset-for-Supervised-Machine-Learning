import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import warnings
 
def svm_classifier_lda():
    warnings.filterwarnings("ignore")
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
    
    print("Applying linear discriminant analysis \n")
    
    #linear discriminant analysis
            
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=6)
    X_train = lda.fit_transform(X_train,y_train)
    X_test = lda.fit_transform(X_test,y_test_0)
    
    print("Applying SVM classifier \n")
    # SVM Classifier
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict_0 = clf.predict(X_test)
    cm = confusion_matrix(y_test_0, y_predict_0)
    print(cm)
    print("Accuracy score of Valence SVM ")
    print(accuracy_score(y_test_0, y_predict_0)*100)
    
    #####################################################3
    y_test_0_file = open("data/y_test_0.dat",'w')
    for i in range(len(y_test_0)):
        y_test_0_file.write(str(y_test_0[i]) + "\n")
    y_test_0_file.close()
        
          
    y_predict_0_file = open("data/y_predict_0.dat",'w')
    for i in range(len(y_test_0)):
        y_predict_0_file.write(str(y_predict_0[i]) + "\n")
    y_predict_0_file.close()
        
    #######################################################################
    
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_1.dat'
    print("\n")
    print("LABEL 1 - Arousal \n ")
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
    # Split the data into training/testing sets
    print("Split the data into training/testing sets \n")
    X_train, X_test, y_train, y_test_1 = train_test_split(X, y, test_size=0.33, random_state=42)
    
      
    # Feature Scaling
    print("Feature Scaling \n")
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #linear discriminant analysis
    print("Applying linear discriminant analysis \n")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=6)
    X_train = lda.fit_transform(X_train,y_train)
    X_test = lda.fit_transform(X_test,y_test_1)
    
        
    # SVM Classifier
    print("Applying SVM classifier \n")
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict_1 = clf.predict(X_test)
    cm = confusion_matrix(y_test_1, y_predict_1)
    print(cm)
    print("Accuracy score of Arousal SVM ")
    print(accuracy_score(y_test_1, y_predict_1)*100)
    
    #######################################################
    y_test_1_file = open("data/y_test_1.dat",'w')
    for i in range(len(y_test_0)):
        y_test_1_file.write(str(y_test_1[i]) + "\n")
        
    y_test_1_file.close()
    
         
          
    y_predict_1_file = open("data/y_predict_1.dat",'w')
    for i in range(len(y_test_0)):
        y_predict_1_file.write(str(y_predict_1[i]) + "\n")
    y_predict_1_file.close()
    
    ##########################################################
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_2.dat'
    print("\n")
    print("LABEL 2 - Dominance \n ")
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
    # Split the data into training/testing sets
    print("Split the data into training/testing sets \n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
      
    # Feature Scaling
    print("Feature Scaling \n")
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #linear discriminant analysis
    print("Applying linear discriminant analysis \n")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=6)
    X_train = lda.fit_transform(X_train,y_train)
    X_test = lda.fit_transform(X_test,y_test)
    
    # SVM Classifier
    print("Applying SVM classifier \n")
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    print("Accuracy score of Dominance SVM ")
    print(accuracy_score(y_test, y_predict)*100)
    
    ##########################################################3
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_3.dat'
    print("\n")
    print("LABEL 3 - Liking \n ")
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
    # Split the data into training/testing sets
    print("Split the data into training/testing sets \n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
       
    # Feature Scaling
    print("Feature Scaling \n")
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #linear discriminant analysis
    print("Applying linear discriminant analysis \n")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=6)
    X_train = lda.fit_transform(X_train,y_train)
    X_test = lda.fit_transform(X_test,y_test)
    
        
    # SVM Classifier
    print("Applying SVM classifier \n")
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    print("Accuracy score of Liking SVM ")
    print(accuracy_score(y_test, y_predict)*100)


if __name__ == '__main__':
    svm_classifier_lda()
