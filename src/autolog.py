import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib



mlflow.set_tracking_uri("http://127.0.0.1:5000")
data=load_wine()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

max_depth=5
n_estimators=6  

# Mention your experiment below
mlflow.set_experiment('exp-autolog')
mlflow.autolog()

with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=2)
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    acc = rf.score(x_test,y_test)

    # mlflow.log_metric('accuracy',acc)
    # mlflow.log_param('max_depth',max_depth)
    # mlflow.log_param('n_estimators',n_estimators)

    #confusion matrix logging thriugh mlflow
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # mlflow.log_artifact("confusion_matrix.png")

    # logging the code
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({'Author':'Bhai','Kaam':'Kuch Nhi'})

    # Logging the model
    model_path='rf_model.pkl'
    joblib.dump(rf,model_path)
    # mlflow.log_artifact(model_path)
    print(acc)
   