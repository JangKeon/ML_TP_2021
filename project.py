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
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# import plotly.express as px


# # Sum of distance for elbow method
# kmeans_elbowDistance = {}

# # For Silhouette
# Kmeans_Sil = {}
# GMM_Sil = {}
# Meanshift_Sil = {}
# DBScan_Sil = {}

# # For Purity
# Kmeans_pur = {}
# GMM_pur = {}
# Meanshift_pur = {}
# DBScan_pur = {}


# def main():
#     # parameter tuning.
#     K_means_parameter = list(range(3, 9, 2))
#     DBScan_parameter = {'eps': [0.1, 0.2, 0.5, 5, 10, 100], 'min_sample': [10, 20, 30]}
#     Meanshift_parameter = [None,0.5, 1.0, 2.0, 10, 100]
#     Meanshift_list = [0, 1.0, 2.0, 10, 100]

#     print("1. Data Load & Preprocessing")
#     data = pd.read_csv('adult.csv')  # load dataset
#     # 1. Change the ? result to NaN
#     data = data.replace('?', np.NaN)

#     # 2. Drop the NaN values (row)
#     data = data.dropna(axis=0)

#     # 3. Drop columns (education, capital-gain, capital-loss)
#     data.drop(['education', 'capital-gain', 'capital-loss'], axis=1, inplace=True)

#     # 4. Change "native-contry" values to binary result
#     # "United-States" : 1, not "United-States" : 0
#     data["native-country"] = data["native-country"].apply(nativeCountry)

#     # 5. Change "income" values to binary result
#     # <=50k : 2, >50k :1
#     data["income"] = data["income"].apply(income)
#     data["income"] = data["income"].apply(pd.to_numeric)

#     # 6. Change educational number to three sector
#     # <10 : 1, 10~13 : 2, >13 :3
#     data["educational-num"] = data["educational-num"].mask(data["educational-num"] < 10, 1)
#     data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 10, 2)
#     data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 11, 2)
#     data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 12, 2)
#     data["educational-num"] = data["educational-num"].mask(data["educational-num"] == 13, 2)
#     data["educational-num"] = data["educational-num"].mask(data["educational-num"] > 13, 3)
#     pd.set_option('display.max_columns', None)

#     print("2. Labeling Income")
#     Income = pd.DataFrame(data["income"])
#     data = data.drop(columns=["income"])
#     Income['label'] = pd.cut(Income["income"], 10)

#     print("3. drop not use data")   ## In process 3 and 4, users can choose what attribute will used in Clustering.
#     data = data.drop(columns=["relationship", "race", "marital-status", "native-country", "workclass", "age", "hours-per-week","gender"])

#     print("4. Divide Scaling list & Encoding List")    ## Except Droped data, Put Encoding attributes and Scaling attributes in turn.
#     pre_feature = Preprocessing(data,
#                                 ["occupation"],
#                                 [ "fnlwgt", "educational-num"])

#     print("5. Make clustering")
#     for preprocess, result in pre_feature.items():
#         Best_Combination(preprocess, result, K_means_parameter, DBScan_parameter, Meanshift_parameter,
#                          Income['label'])

#     print("6. Result")
#     ## Check sum of distance for elbow method
#     ShowPlot("KMeans_distance", kmeans_elbowDistance, K_means_parameter)

#     ## Silhouette
#     ShowPlot("KMeans_silhouette", Kmeans_Sil, K_means_parameter)
#     ShowPlot("EM_silhouette", GMM_Sil, K_means_parameter)
#     ShowPlot("DBSCAN_silhouette", DBScan_Sil, DBScan_parameter['eps'])
#     ShowPlot("MeanShift_distance", Meanshift_Sil, Meanshift_list)

#     preprocess, result = FindBestResult(Kmeans_Sil)
#     print("K-means best silhouette : ", result, preprocess)
#     preprocess, result = FindBestResult(GMM_Sil)
#     print("EM best silhouette : ", result, preprocess)
#     preprocess, result = FindBestResult(DBScan_Sil)
#     print("DBSCAN best silhouette : ", result, preprocess)
#     preprocess, result = FindBestResult(Meanshift_Sil)
#     print("MeanShift best silhouette : ", result, preprocess)

#     # purity
#     ShowPlot("KMeans_purity", Kmeans_pur, K_means_parameter)
#     ShowPlot("EM_purity", GMM_pur, K_means_parameter)
#     ShowPlot("DBSCAN_purity", DBScan_pur, DBScan_parameter['eps'])
#     ShowPlot("MeanShift_purity", Meanshift_pur, Meanshift_list)

#     preprocess, result = FindBestResult(Kmeans_pur)
#     print("K-means best purity : ", result, preprocess)
#     preprocess, result = FindBestResult(GMM_pur)
#     print("K-means best purity : ", result, preprocess)
#     preprocess, result = FindBestResult(DBScan_pur)
#     print("DBSCAN best purity : ", result, preprocess)
#     preprocess, result = FindBestResult(Meanshift_pur)
#     print("MeanShift best purity : ", result, preprocess)


# def income(x):
#     if x == '<=50K':
#         return x.replace(x, "0")
#     return "1"


# def nativeCountry(x):
#     if x != "United-States":
#         return str(x).replace(str(x), "0")
#     return "1"


# # for one-hot-encoding
# def dummy_data(data, columns):
#     for column in columns:
#         data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
#         data = data.drop(column, axis=1)
#     return data


# def Preprocessing(feature, encode_list, scale_list):
#     # feature : dataframe of feature

#     # scaler
#     scaler_stndard = preprocessing.StandardScaler()
#     scaler_MM = preprocessing.MinMaxScaler()
#     scaler_robust = preprocessing.RobustScaler()
#     scaler_maxabs = preprocessing.MaxAbsScaler()
#     scaler_normalize = preprocessing.Normalizer()
#     scalers = [None, scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]
#     scalers_name = ["original", "standard", "minmax", "robust", "maxabs", "normalize"]

#     # encoder
#     encoder_ordinal = preprocessing.OrdinalEncoder()
#     # one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
#     encoders_name = ["ordinal", "ordinal"]

#     # result box
#     result_dictionary = {}
#     i = 0

#     if encode_list == []:
#         for scaler in scalers:
#             if i == 0:  # not scaling
#                 result_dictionary[scalers_name[i]] = feature.copy()

#             else:
#                 # ===== scalers
#                 result_dictionary[scalers_name[i]] = feature.copy()
#                 result_dictionary[scalers_name[i]][scale_list] = scaler.fit_transform(feature[scale_list])  # scaling
#             i = i + 1
#         return result_dictionary

#     for scaler in scalers:
#         if i == 0:  # not scaling
#             result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
#                 feature[encode_list])
#             result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
#                                                                         encode_list)

#         else:
#             # ===== scalers + ordinal encoding
#             result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_ordinal"][scale_list] = scaler.fit_transform(
#                 feature[scale_list])  # scaling
#             result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
#                 feature[encode_list])  # encoding

#             # ===== scalers + OneHot encoding
#             result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
#             result_dictionary[scalers_name[i] + "_onehot"][scale_list] = scaler.fit_transform(
#                 feature[scale_list])  # scaling
#             result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
#                                                                         encode_list)  # encoding

#         i = i + 1

#     return result_dictionary


# def Best_Combination(preprocessing_name, feature, n_cluster, DBSCAN_list, MeanShift_list, purity_GT):
#     print(preprocessing_name)
#     print(feature)
#     print(feature.columns)
#     # n_cluster : list number of cluster (use in Kmeans, GMM)
#     # DBSCAN_list : list of DBSCAN parameters (eps, min_sample)
#     # MeanShift_list : list of MeanShift parameters (bandwidth)

#     pca = PCA(n_components=2)
#     clms = feature.columns

#     KMeans
#     print("Kmeans")
#     Kmean_Distance = []
#     Kmeans_Sil_result = []
#     Kmeans_pur_result = []

#     for k in n_cluster:
#         df_feature_pca = feature[clms]
#         df_feature_pca = pca.fit_transform(df_feature_pca)
#         df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])

#         # arr = df_feature_pca[["PC1", "PC2"]]
#         kmeans = KMeans(n_clusters=k).fit(df_feature_pca)
#         # sum of distance for elbow methods
#         Kmean_Distance.append(kmeans.inertia_)
#         # silhouette (range -1~1)
#         Kmeans_Sil_result.append(silhouette_score(df_feature_pca, kmeans.labels_, metric='euclidean'))
#         # purity
#         Kmeans_pur_result.append(purity_score(purity_GT, kmeans.labels_))
#         label = kmeans.labels_
#         # Visualization
#         fig = px.scatter(
#             df_feature_pca,
#             x=df_feature_pca["PC1"],
#             y=df_feature_pca["PC2"],
#             color=label,
#             title="KMeans"
#         )
#         # fig.show()

#     kmeans_elbowDistance[preprocessing_name] = Kmean_Distance
#     Kmeans_Sil[preprocessing_name] = Kmeans_Sil_result
#     Kmeans_pur[preprocessing_name] = Kmeans_pur_result

#     print("EM")
#     GMM_Sil_result = []
#     GMM_Pur_result = []

#     for k in n_cluster:
#         # Use PCA, visualization
#         df_feature_pca = feature[clms]
#         df_feature_pca = pca.fit_transform(df_feature_pca)
#         df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])

#         gmm = GaussianMixture(n_components=k)
#         labels = gmm.fit_predict(df_feature_pca)

#         # silhouette
#         GMM_Sil_result.append(silhouette_score(df_feature_pca, labels, metric='euclidean'))

#         # purity
#         GMM_Pur_result.append(purity_score(purity_GT, labels))

#         fig = px.scatter(
#                     df_feature_pca,
#                     x=df_feature_pca["PC1"],
#                     y=df_feature_pca["PC2"],
#                     color=labels,
#                     title="EM"
#                 )
#         # fig.show()

#     GMM_Sil[preprocessing_name] = GMM_Sil_result
#     GMM_pur[preprocessing_name] = GMM_Pur_result


#     # DBSCAN
#     print("DBScan")
#     DBScan_Sil_result = []
#     DBScan_Pur_result = []

#     for eps in DBSCAN_list["eps"]:
#         max_silhouette = -2
#         max_purity = -2

#         for DBS in DBSCAN_list["min_sample"]:
#             df_feature_pca = feature[clms]
#             df_feature_pca = pca.fit_transform(df_feature_pca)
#             df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])

#             dbscan = DBSCAN(eps=eps, min_samples=DBS)
#             label = dbscan.fit_predict(df_feature_pca)

#             fig = px.scatter(
#                 df_feature_pca,
#                 x=df_feature_pca["PC1"],
#                 y=df_feature_pca["PC2"],
#                 color=label,
#                 title="DBSCAN"
#             )
#             # fig.show()

#             # silhouette (range -1~1)
#             try:
#                 current_silhouette = silhouette_score(feature, label, metric='euclidean')
#             except:
#                 current_silhouette = -5

#             if max_silhouette < current_silhouette:
#                 max_silhouette = current_silhouette

#             # purity
#             current_purity = purity_score(purity_GT, label)
#             if max_purity < current_purity:
#                 max_purity = current_purity

#         DBScan_Sil_result.append(max_silhouette)
#         DBScan_Pur_result.append(max_purity)

#     DBScan_Sil[preprocessing_name] = DBScan_Sil_result
#     DBScan_pur[preprocessing_name] = DBScan_Pur_result

#     # MeanShift
#     print("Meanshift")
#     Meanshift_Sil_result = []
#     Meanshift_Pur_result = []

#     for MS in MeanShift_list:
#         meanShift = MeanShift(bandwidth=MS)
#         label = meanShift.fit_predict(feature)

#         df_feature_pca = feature[clms]
#         df_feature_pca = pca.fit_transform(df_feature_pca)
#         df_feature_pca = pd.DataFrame(df_feature_pca, columns=["PC1", "PC2"])


#         label = meanShift.fit_predict(df_feature_pca)

#         fig = px.scatter(
#             df_feature_pca,
#             x=df_feature_pca["PC1"],
#             y=df_feature_pca["PC2"],
#             color=label,
#             title="MeanShift"
#         )
#         # fig.show()

#         # silhouette (range -1~1)
#         try:
#             current_silhouette = silhouette_score(feature, label, metric='euclidean')
#         except:
#             current_silhouette = -1
#         # silhouette (range -1~1)
#         Meanshift_Sil_result.append(current_silhouette)

#         # purity
#         Meanshift_Pur_result.append(purity_score(purity_GT, label))

#     Meanshift_Sil[preprocessing_name] = Meanshift_Sil_result
#     Meanshift_pur[preprocessing_name] = Meanshift_Pur_result


# # Test purity
# def purity_score(y_true, y_pred):
#     # compute contingency matrix (also called confusion matrix)
#     contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
#     # return purity
#     return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# def ShowPlot(title, dict, x_list):
#     for key, value in dict.items():
#         plt.plot(x_list, value, label=key)

#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
#     plt.tight_layout()
#     plt.show()


# def ResultPrint(x):
#     result = {
#         'sum': x.sum(),
#         'count': x.count(),
#         'mean': x.mean(),
#         'variance': x.var()
#     }
#     return result


# def FindBestResult(dict):
#     key = None
#     largest = 0
#     for keys, item in dict.items():
#         if max(item) > largest:
#             largest = max(item)
#             key = keys

#     return key, largest


# if __name__ == "__main__":
#     main()
