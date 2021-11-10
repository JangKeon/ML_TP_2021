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



# import pandas as pd
# import numpy as np
# import matplotlib.pylab as plt
# from sklearn import preprocessing, metrics
# from sklearn.cluster import KMeans, MeanShift, DBSCAN
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score
# import time
# 
# # sum of distance for elbow method
# kmeans_sumofDistance = {}
# 
# # silhouette
# kmeans_silhouette = {}
# gmm_silhouette = {}
# meanshift_silhouette = {}
# dbscan_silhouette = {}
# 
# # purity
# kmeans_purity = {}
# gmm_purity = {}
# meanshift_purity = {}
# dbscan_purity = {}
# 
# 
# def main():
#     # hyperparameter
#     n_cluster = list(range(4, 6, 1))
#     DBSCAN_list = {'eps': [0.1, 0.2, 0.5, 5], 'min_sample': [10, 20]}
#     MeanShift_list = [None, 1.0, 2.0, 10, 20]
#     MeanShift_list_plot = [0, 1.0, 2.0, 10, 20]
# 
#     print("1. Data Load & Pre processing")
#     dataset = pd.read_csv('adult.csv')  # load dataset
#     # 1. Change the ? value to NaN
#     dataset = dataset.replace('?', np.NaN)
# 
#     # 2. Drop the NaN values (row)
#     dataset = dataset.dropna(axis=0)
# 
#     # 3. Drop columns (education, capital-gain, capital-loss)
#     dataset.drop(['education', 'capital-gain', 'capital-loss'], axis=1, inplace=True)
# 
#     # 4. Change "native-contry" values to binary value
#     # "United-States" : 1, not "United-States" : 0
#     dataset["native-country"] = dataset["native-country"].apply(nativeCountry)
# 
#     # 5. Change "income" values to binary value
#     # <=50k : 2, >50k :1
#     dataset["income"] = dataset["income"].apply(income)
#     dataset["income"] = dataset["income"].apply(pd.to_numeric)
#     # 6. Change educational number to three sector
#     # <10 : 1, 10~13 : 2, >13 :3
#     dataset["educational-num"] = dataset["educational-num"].mask(dataset["educational-num"] < 10, 1)
#     dataset["educational-num"] = dataset["educational-num"].mask(dataset["educational-num"] == 10, 2)
#     dataset["educational-num"] = dataset["educational-num"].mask(dataset["educational-num"] == 11, 2)
#     dataset["educational-num"] = dataset["educational-num"].mask(dataset["educational-num"] == 12, 2)
#     dataset["educational-num"] = dataset["educational-num"].mask(dataset["educational-num"] == 13, 2)
#     dataset["educational-num"] = dataset["educational-num"].mask(dataset["educational-num"] > 13, 3)
# 
# 
#     pd.set_option('display.max_columns', None)
#     print(dataset)
# 
#     print("2. Labeling Income")
#     Income = pd.DataFrame(dataset["income"])
#     dataset = dataset.drop(columns=["income"])
#     Income['label'] = pd.cut(Income["income"], 10)
# 
#     print("3. drop not use data")
#     dataset = dataset.drop(columns=["workclass", "relationship", "race", "occupation",  "gender", "native-country"])
# 
#     print("4. Divide Scaling list & Encoding List")
#     pre_feature = Preprocessing(dataset,
#                                 ["marital-status"],
#                                 ["age", "fnlwgt", "educational-num", "hours-per-week"])
#     # pprint.pprint(pre_feature)
# 
#     print("5. make clustering")
#     for key, value in pre_feature.items():
#         FindBestCombination(key, value, n_cluster, DBSCAN_list, MeanShift_list,
#                             Income['label'])
# 
#     print("=== 6. Result")
#     # check sum of distance for elbow method
#     makeplot("KMeans_distance", kmeans_sumofDistance, n_cluster)
#     # #silhouette score
#     makeplot("KMeans_silhouette", kmeans_silhouette, n_cluster)
#     makeplot("EM_silhouette", gmm_silhouette, n_cluster)
#     makeplot("DBSCAN_silhouette", dbscan_silhouette, DBSCAN_list['eps'])
#     makeplot("MeanShift_distance", meanshift_silhouette, MeanShift_list_plot)
# 
# 
# 
# 
#     key, value = fineMaxValueKey(kmeans_silhouette)
#     print("k-means best silhouette : ", value, key)
#     key, value = fineMaxValueKey(gmm_silhouette)
#     print("EM best silhouette : ", value, key)
#     key, value = fineMaxValueKey(dbscan_silhouette)
#     print("DBSCAN best silhouette : ", value, key)
#     key, value = fineMaxValueKey(meanshift_silhouette)
#     print("MeanShift best silhouette : ", value, key)
# 
#     # purity
#     makeplot("KMeans_purity", kmeans_purity, n_cluster)
#     makeplot("EM_purity", gmm_purity, n_cluster)
#     makeplot("DBSCAN_purity", dbscan_purity, DBSCAN_list['eps'])
#     makeplot("MeanShift_purity", meanshift_purity, MeanShift_list_plot)
# 
#     key, value = fineMaxValueKey(kmeans_purity)
#     print("k-means best purity : ", value, key)
#     key, value = fineMaxValueKey(gmm_purity)
#     print("k-means best purity : ", value, key)
#     key, value = fineMaxValueKey(dbscan_purity)
#     print("DBSCAN best purity : ", value, key)
#     key, value = fineMaxValueKey(meanshift_purity)
#     print("MeanShift best purity : ", value, key)
# 
# 
# def income(x):
#     if x == '<=50K':
#         return x.replace(x, "0")
#     return "1"
# 
# 
# def nativeCountry(x):
#     if x != "United-States":
#         return str(x).replace(str(x), "0")
#     return "1"
# 
# 
# # for one-hot-encoding
# def dummy_data(data, columns):
#     for column in columns:
#         data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
#         data = data.drop(column, axis=1)
#     return data
# 
# 
# def Preprocessing(feature, encode_list, scale_list):
#     # feature : dataframe of feature
# 
#     # scaler
#     scaler_stndard = preprocessing.StandardScaler()
#     scaler_MM = preprocessing.MinMaxScaler()
#     scaler_robust = preprocessing.RobustScaler()
#     scaler_maxabs = preprocessing.MaxAbsScaler()
#     scaler_normalize = preprocessing.Normalizer()
#     scalers = [None, scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]
#     scalers_name = ["original", "standard", "minmax", "robust", "maxabs", "normalize"]
# 
#     # encoder
#     encoder_ordinal = preprocessing.OrdinalEncoder()
#     # one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
#     encoders_name = ["ordinal", "onehot"]
# 
#     # result box
#     result_dictionary = {}
#     i = 0
# 
#     if encode_list == []:
#         for scaler in scalers:
#             if i == 0:  # not scaling
#                 result_dictionary[scalers_name[i]] = feature.copy()
# 
#             else:
#                 # ===== scalers
#                 result_dictionary[scalers_name[i]] = feature.copy()
#                 result_dictionary[scalers_name[i]][scale_list] = scaler.fit_transform(feature[scale_list])  # scaling
#             i = i + 1
#         return result_dictionary
# 
#     for scaler in scalers:
#         if i == 0:  # not scaling
#             result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
#                 feature[encode_list])
#             result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
#                                                                         encode_list)
# 
#         else:
#             # ===== scalers + ordinal encoding
#             result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_ordinal"][scale_list] = scaler.fit_transform(
#                 feature[scale_list])  # scaling
#             result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
#                 feature[encode_list])  # encoding
# 
#             # ===== scalers + OneHot encoding
#             result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_onehot"][scale_list] = scaler.fit_transform(
#                 feature[scale_list])  # scaling
#             result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
#                                                                         encode_list)  # encoding
# 
#         i = i + 1
# 
#     return result_dictionary
# 
# 
# def FindBestCombination(preprocessing_name, feature, n_cluster, DBSCAN_list, MeanShift_list, purity_GT):
#     print(preprocessing_name)
#     # n_cluster : list number of cluster (use in Kmeans, GMM)
#     # DBSCAN_list : list of DBSCAN parameters (eps, min_sample)
#     # MeanShift_list : list of MeanShift parameters (bandwidth)
# 
#     # KMeans
#     print("Kmeans")
#     start_time = time.time()
#     kmean_sum_of_squared_distances = []
#     kmean_silhouette_sub = []
#     kmeans_purity_sub = []
# 
#     for k in n_cluster:
#         kmeans = KMeans(n_clusters=k).fit(feature)
#         # sum of distance for elbow methods
#         kmean_sum_of_squared_distances.append(kmeans.inertia_)
#         # silhouette (range -1~1)
#         kmean_silhouette_sub.append(silhouette_score(feature, kmeans.labels_, metric='euclidean'))
#         # purity
#         kmeans_purity_sub.append(purity_score(purity_GT, kmeans.labels_))
# 
#     kmeans_sumofDistance[preprocessing_name] = kmean_sum_of_squared_distances
#     kmeans_silhouette[preprocessing_name] = kmean_silhouette_sub
#     kmeans_purity[preprocessing_name] = kmeans_purity_sub
#     print(time.time() - start_time)
# 
#     # GaussianMixture (EM, GMM)
#     print("EM")
#     start_time = time.time()
#     gmm_silhouette_sub = []
#     gmm_purity_sub = []
# 
#     for k in n_cluster:
#         gmm = GaussianMixture(n_components=k)
#         labels = gmm.fit_predict(feature)
# 
#         # silhouette (range -1~1)
#         gmm_silhouette_sub.append(silhouette_score(feature, labels, metric='euclidean'))
# 
#         # purity
#         gmm_purity_sub.append(purity_score(purity_GT, labels))
# 
#     gmm_silhouette[preprocessing_name] = gmm_silhouette_sub
#     gmm_purity[preprocessing_name] = gmm_purity_sub
#     print(time.time() - start_time)
# 
#     # DBSCAN
#     print("dbscan")
#     start_time = time.time()
#     dbscan_silhouette_sub = []
#     dbscan_purity_sub = []
# 
#     for eps in DBSCAN_list["eps"]:
#         max_silhouette = -2
#         max_purity = -2
# 
#         for sample in DBSCAN_list["min_sample"]:
#             dbscan = DBSCAN(eps=eps, min_samples=sample)
#             label = dbscan.fit_predict(feature)
# 
#             # silhouette (range -1~1)
#             try:
#                 current_silhouette = silhouette_score(feature, label, metric='euclidean')
#             except:
#                 current_silhouette = -5
# 
#             if max_silhouette < current_silhouette:
#                 max_silhouette = current_silhouette
# 
#             # purity
#             current_purity = purity_score(purity_GT, label)
#             if max_purity < current_purity:
#                 max_purity = current_purity
# 
#         dbscan_silhouette_sub.append(max_silhouette)
#         dbscan_purity_sub.append(max_purity)
# 
#     dbscan_silhouette[preprocessing_name] = dbscan_silhouette_sub
#     dbscan_purity[preprocessing_name] = dbscan_purity_sub
#     print(time.time() - start_time)
# 
#     # meanShift
#     print("meanshift")
#     start_time = time.time()
#     meanshift_silhouette_sub = []
#     meanshift_purity_sub = []
# 
#     for bw in MeanShift_list:
#         meanShift = MeanShift(bandwidth=bw)
#         label = meanShift.fit_predict(feature)
#         print(label)
#         print(time.time() - start_time)
# 
#         # silhouette (range -1~1)
#         try:
#             current_silhouette = silhouette_score(feature, label, metric='euclidean')
#         except:
#             current_silhouette = -1
#         # silhouette (range -1~1)
#         meanshift_silhouette_sub.append(current_silhouette)
# 
#         # purity
#         meanshift_purity_sub.append(purity_score(purity_GT, label))
# 
#     meanshift_silhouette[preprocessing_name] = meanshift_silhouette_sub
#     meanshift_purity[preprocessing_name] = meanshift_purity_sub
#     print(time.time() - start_time)
# 
# 
# # Test purity
# def purity_score(y_true, y_pred):
#     # compute contingency matrix (also called confusion matrix)
#     contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
#     # return purity
#     return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
# 
# 
# def makeplot(title, dict, x_list):
#     for key, value in dict.items():
#         plt.plot(x_list, value, label=key)
# 
#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
#     plt.tight_layout()
#     plt.show()
# 
# 
# def my_summary(x):
#     result = {
#         'sum': x.sum(),
#         'count': x.count(),
#         'mean': x.mean(),
#         'variance': x.var()
#     }
#     return result
# 
# 
# def fineMaxValueKey(dict):
#     key = None
#     largest = 0
#     for keys, item in dict.items():
#         if max(item) > largest:
#             largest = max(item)
#             key = keys
# 
#     return key, largest
# 
# 
# if __name__ == "__main__":
#     main()
