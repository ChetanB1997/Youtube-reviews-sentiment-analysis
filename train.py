from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


from sentiment_a import sampling


def model(Xdata,Ydata):
    
    X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.3, random_state=0)
    model = GaussianNB()
    model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    print(y_test,y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    nb_score = accuracy_score(y_test, y_pred)
    #model.save_weights("sentiment_model.h5")
       
    return cm,nb_score

