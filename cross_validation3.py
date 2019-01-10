import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def cross_validate3():
    
    # Get Data
    file_x = 'data/features_sampled.dat'
    file_y = 'data/label_class_3.dat'
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
   #X = StandardScaler().fit_transform(X)
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
     # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #principle component analysis
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    
   
    	
    models = []
    models.append(('LR', LogisticRegression(random_state = 42)))
    models.append(('SVC', SVC(kernel = 'rbf', random_state = 42)))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
    models.append(('DT', DecisionTreeClassifier(random_state = 42)))
    
    scoring = 'accuracy'
    
    # Cross Validate
    results = []
    names = []
    timer = []
    print('Model | Mean of CV | Std. Dev. of CV | Time')
    for name, model in models:
        start_time = time.time()
        kfold = model_selection.KFold(n_splits=5, random_state=42)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        t = (time.time() - start_time)
        timer.append(t)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f) %f s" % (name, cv_results.mean(), cv_results.std(), t)
        print(msg)
    
    models = []
    for i in range(1,41): 
        models.append(('KNN', KNeighborsClassifier(n_neighbors=i)))
        results = []
        names = []
        timer = []
        cv_knn = []
        
        print('Model  | Mean of CV | Std. Dev. of CV | Time',i)
        for name, model in models:
            start_time = time.time()
            kfold = model_selection.KFold(n_splits=5, random_state=42)
            cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            t = (time.time() - start_time)
            timer.append(t)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f) %f s" % (name, cv_results.mean(), cv_results.std(), t)
                
            cv_knn.append(cv_results.mean())
        print(msg)
    print('\nmaximum accuracy for liking is',max(cv_knn))
    plt.figure(figsize=(12, 6))  
    plt.plot(range(1, 41), cv_knn, color='red', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=10)
    plt.title('mean cv_results of  K Value for liking')  
    plt.xlabel('K Value')  
    plt.ylabel('Mean cv_result')  
        
if __name__ == '__main__':
    cross_validate3()