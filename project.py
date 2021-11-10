import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# ----------------------------------- Functions -----------------------------------
def nativeCountry(x) :
  if x != "United-States" :
    return str(x).replace(str(x),"0")
  return "1"

def income(x) :
  if x == '<=50K' :
    return x.replace(x,"0")
  return "1"


def oneHotEncode_category(arr, col_list):
  enc = preprocessing.OneHotEncoder()
  for col in col_list:
    encodedData = enc.fit_transform(arr[[col]])
    encodedDataRecovery = np.argmax(encodedData, axis=1).reshape(-1, 1)
    arr[col] = encodedDataRecovery

def ordinalEncode_category(df, col_list):
  ordinalEncoder = preprocessing.OrdinalEncoder()
  for col in col_list:
    X = pd.DataFrame(df[col])
    ordinalEncoder.fit(X)
    df[col] = pd.DataFrame(ordinalEncoder.transform(X))


def encodingNscalingData(dataset, scaled_col, encoded_col):
  result = []
  for encoder in [0, 1]:
    new_df = dataset.copy();
    if encoder == 0:
      ordinalEncode_category(new_df, encoded_col)
    elif encoder == 1:
      ordinalEncode_category(new_df, encoded_col)
    new_df.dropna(axis=0, inplace=True)
    for scaler in [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]:
      for col in scaled_col:
        new_df[col] = scaler.fit_transform(new_df[col].values[:, np.newaxis]).reshape(-1)
      result.append(new_df)
  return result

def callClassificationParameters(num):
  decisionTreepParameters = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [3, 5, 10],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [3, 5, 2],
    "max_features": ["auto", "sqrt", "log2"]
  }
  logisticRegressionParameters = {
    "penalty": ['l2', 'l1'],
    "solver": ['saga', "liblinear"],
    "multi_class": ['auto', 'ovr'],
    "random_state": [3, 5, 10],
    "C": [1.0, 0.5],
    "max_iter": [1000]
  }
  svmParameters = {
    "decision_function_shape": ['ovo', 'ovr'],
    "gamma": ['scale', 'auto'],
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "C": [1.0, 0.5]
  }
  if num == 0:
    return logisticRegressionParameters
  elif num == 1:
    return decisionTreepParameters
  else:
    return svmParameters

def printConfusionMatrix(model, x, y):
  em = model.best_estimator_
  pred = em.predict(x)
  
  # (optional) accuracy score
  accs = accuracy_score(y, pred)
  print("accuracy_score", accs)
  
  # Confusiton Matrix
  cf = confusion_matrix(y, pred)
  print(cf)

def findBestClassificationModel(data_list,target):
  bestscore = -1
  i = 0
  best_model = None
  best_test_set = None
  for dataset in data_list:
    i = 0
    y = dataset[target]
    x = dataset.drop([target], axis=1)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=0)
    for model in [LogisticRegression(), DecisionTreeClassifier(), svm.SVC()]:
      tunedModel = GridSearchCV(model, callClassificationParameters(i), scoring='neg_mean_squared_error', cv=5)
      tunedModel.fit(train_x, train_y)
      print("-------------------------------")
      print(tunedModel.best_params_)
      print(tunedModel.best_score_)
      printConfusionMatrix(tunedModel, test_x, test_y)
      i = i + 1
      if bestscore < tunedModel.best_score_:
        bestscore = tunedModel.best_score_
        bestparams = tunedModel.best_params_
        bestscore = tunedModel.best_score_
        bestparams = tunedModel.best_params_
        # save best model and test set
        best_model = tunedModel
        best_test_set = (test_x, test_y)

  print("------Best model-------")
  print(bestparams)
  print(bestscore)
  return best_model, best_test_set # return best model and test set

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='Best model')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


# Load the dataset (adult.csv)
df = pd.read_csv('adult.csv')

# ----------------------------------- Preprocessing -----------------------------------
# 1. Change the ? value to NaN
df = df.replace('?', np.NaN)

# 2. Drop the NaN values (row)
df = df.dropna(axis=0)

# 3. Drop columns (education, capital-gain, capital-loss)
df.drop(['education', 'capital-gain', 'capital-loss'], axis=1, inplace=True)

# 4. Change "native-contry" values to binary value
# "United-States" : 1, not "United-States" : 0
df["native-country"]=df["native-country"].apply(nativeCountry)

# 5. Change "income" values to binary value
# <=50k : 2, >50k :1
df["income"]=df["income"].apply(income)

# 6. Change educational number to three sector
# <10 : 1, 10~13 : 2, >13 :3
df["educational-num"]=df["educational-num"].mask(df["educational-num"] < 10, 1)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 10, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 11, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 12, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 13, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] > 13, 3)

pd.set_option('display.max_columns', None)
print(df)


scaled_col=["age","fnlwgt","educational-num","hours-per-week","native-country","income"]
encoded_col=["workclass","marital-status","occupation","relationship","race","gender"]
preprocessing_list=encodingNscalingData(df, scaled_col, encoded_col)
print(preprocessing_list)

# print(findBestClassificationModel(preprocessing_list,"income"))

model, (test_x, test_y) = findBestClassificationModel(preprocessing_list,"income")
confusionMatrix(model, test_x, test_y)

# visualize ROC curve
prob = model.predict_proba(test_x)
prob = prob[:, 1]
fper, tper, thresholds = roc_curve(test_y, prob)
plot_roc_curve(fper, tper)

# visualize confusion matrix
label = ['0', '1']
plot = plot_confusion_matrix(model, test_x, test_y, display_labels=label, cmap=plt.cm.Blues, normalize=None)
plot.ax_.set_title('Confusion Matrix')

