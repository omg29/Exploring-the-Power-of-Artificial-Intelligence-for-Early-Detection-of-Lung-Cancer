import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Replace 'file_path.csv' with the actual file path to your CSV file
file_path = r'C:\Users\gagrv\Desktop\LungCancerDataset20240914.csv'

# Reading the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

df['GENDER'] = df['GENDER'].map({'F': 0, 'M': 1})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

#target column (model will predict this column)
Y = df['LUNG_CANCER'] #Target colmun (Output from model)

X = df.drop('LUNG_CANCER', axis=1) # All other columns except for target column (Input provided to model)

#Spilt the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

#Define the model
#regr = linear_model.LinearRegression()
clf = MLPClassifier(max_iter=80).fit(X_train, y_train)
DecisionTree = DecisionTreeClassifier(max_depth=80).fit(X_train, y_train)
RandomForestClassifier = RandomForestClassifier(max_depth=90.000).fit(X_train, y_train)
LogisticsRegrssion = LogisticRegression().fit(X_train, y_train)



#Train
#DecisionTree.fit(X_train, y_train)

#Prediction
clf_prediction = clf.predict(X_test)
print("Accuracy score of clf test:", str(accuracy_score(y_test, clf_prediction)))
clf_prediction_train = clf.predict(X_train)
print("Accuracy score of clf train:", str(accuracy_score(y_train, clf_prediction_train)))

#CLF_sum = int(clf_prediction) + int(clf_prediction_train)
#print("Accuracy of CLF", (clf_prediction/CLF_sum) + (clf_prediction_train/CLF_sum))

DecisionTree_Prediction = DecisionTree.predict(X_test)
print("Accuracy score of Decision Tree test:", str(accuracy_score(y_test, DecisionTree_Prediction)))
DecisionTree_Prediction_Train = DecisionTree.predict(X_train)
print("Accuracy score of Decision Tree train:", str(accuracy_score(y_train, DecisionTree_Prediction_Train)))

RandomForestClassifier_Prediction = RandomForestClassifier.predict(X_test)
print("Accuracy score of Random Forest test:", str(accuracy_score(y_test, RandomForestClassifier_Prediction)))
RandomForestClassifier_Prediction_Train= RandomForestClassifier.predict(X_train)
print("Accuracy score of Random Forest train:", str(accuracy_score(y_train, RandomForestClassifier_Prediction_Train)))

LogisticRegression_Prediction = LogisticsRegrssion.predict(X_test)
print("Accuracy score of Logistics Regression test:", str(accuracy_score(y_test, LogisticRegression_Prediction)))
LogisticRegression_Prediction_Train = LogisticsRegrssion.predict(X_train)
print("Accuracy score of Logistics Regression train:", str(accuracy_score(y_train, LogisticRegression_Prediction_Train)))

#Evaluation
#print("Mean squared error: %.2f" % mean_squared_error(y_test, linear_prediction))

#print("Coefficients: \n", regr.coef_)

#Combine Models
w_clf = 10
w_tree = 10
w_forest = 10
w_regression = 10

print(type(y_test))

for i in range (len(X_test)):

    #clf
    if clf_prediction[i] != y_test.iloc[i]:
        w_clf /= 2
    else:
        w_clf *= 2

    #tree
    if DecisionTree_Prediction[i] != y_test.iloc[i]:
        w_tree /= 2
    else:
        w_tree *= 2

    #forest
    if RandomForestClassifier_Prediction[i] != y_test.iloc[i]:
        w_forest /= 2
    else:
        w_forest *= 2

    #regresison
    if LogisticRegression_Prediction[i] != y_test.iloc[i]:
        w_regression /= 2
    else:
        w_regression *= 2

    weights_sum = w_clf + w_tree + w_forest + w_regression
    w_clf /= weights_sum
    w_tree /= weights_sum
    w_forest /= weights_sum
    w_regression /= weights_sum

    #print(w_clf + w_tree + w_forest + w_regression)

print(" ")
print("CLF Final Weight:", w_clf)
print("Decision Tree Final Weight:", w_tree)
print("Random Forest Final Weight:", w_forest)
print("Logistics Regression Final Weight:", w_regression)


counter_0 = 0
counter_1 = 0
final_decision = None

"""
def test(counter_0, counter_1, final_decision):
    
    if clf_prediction[a] == 0:
        counter_0 += 1
    else:
        counter_1 += 1

    if DecisionTree_Prediction[a] == 0:
        counter_0 += 1
    else:
        counter_1 += 1

    if RandomForestClassifier_Prediction[a] == 0:
        counter_0 += 1
    else:
        counter_1 += 1
    
    if LogisticRegression_Prediction[a] == 0:
        counter_0 += 1
    else:
        counter_1 += 1

    if counter_0 > counter_1:
        final_decision = "No"
    else:
        final_decision = "Yes"

    print(final_decision)

test(counter_0, counter_1, final_decision)
"""