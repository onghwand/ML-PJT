# 중고차 가격 예측(RF)

> ## P1-1. Environmental Set-up & Data Loading

```python
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# # enter the foldername in your Google Drive where you have saved the unzipped
# FOLDERNAME =  'ADX/'

# assert FOLDERNAME is not None, 'ERROR'

%cd drive/My\ Drive
%cp -r $FOLDERNAME ../../

# 한글 나눔포트 사용
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

```python
import pandas as pd
import numpy as np
from time import time
import datetime
import math

from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='NanumBarunGothic')

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
```

```python
# 모델 수립을 위한 Train/Validation Set
df = pd.read_csv('./ADX/Data_UsedCarPrediction/P1_dataset.csv', encoding ='cp949')  # 한글 Encoding 문제로 encoding = 'cp949'

# 시스템 구현을 위한 Test Set 샘플
df_test = pd.read_csv('./ADX/Data_UsedCarPrediction/P1_testset_sample.csv', encoding ='cp949')  # 한글 Encoding 문제로 encoding = 'cp949'
```

```python
#df.head()
#df.shape
for col in df.columns:
  val=df[col].isnull().sum()
  if val>0:
    print(col,df[col].isnull().sum())
```

```python
df_view=df[['NC_GRADE_PRICE','SHIPPING_PRICE','EXHA','NAVIGATION','NEWCARPRIC','NC_GRADE_KEY','SMARTKEY','VDC','DT_MODEL_KEY']]
plt.figure(figsize=(20,10))
for i in range(0,9):
    plt.subplot(3,3,i+1)
    plt.scatter(df_view.iloc[:,i], df_view['SHIPPING_PRICE'])
    plt.title(df_view.columns[i])
```

> ## P1-2. Modelling
>
> - df를 사용하여 예측모형 수립
> - 상세 사항은 업로드 된 비디오 참고

```python
def preprocessing_EXHA(df):
    return_df = df.copy()
    # EXHA 결측값 제거
    return_df["FUELNM"] = return_df["FUELNM"].fillna("global")
    exha_dict = dict(return_df["EXHA"].groupby(return_df["FUELNM"]).mean())
    exha_dict["global"] = return_df["EXHA"].mean()
    return_df["EXHA_ADJ"] = return_df.apply(lambda x: exha_dict[x["FUELNM"]]  if x["EXHA"] < 10 or np.isnan(x["EXHA"]) else x["EXHA"], axis=1) 
    return return_df
df_1 = preprocessing_EXHA(df)

return_df = df_1.copy()
price_list = []
return_df["EXHA_THOUSAND"] = return_df["EXHA_ADJ"].apply(lambda x: (x-1)//1000)  

def preprocessing_PRICE(df):
    return_df = df.copy()
    price_list = []
    return_df["EXHA_THOUSAND"] = return_df["EXHA_ADJ"].apply(lambda x: (x-1)//1000)    
    exha_dict = dict(return_df["SHIPPING_PRICE"].groupby(return_df["EXHA_THOUSAND"]).mean().round())
    global_mean = return_df["SHIPPING_PRICE"].mean()

    for shipping_price, nc_grade_price, new_car_price, exha_thousand in return_df[["SHIPPING_PRICE", "NC_GRADE_PRICE", "NEWCARPRIC", "EXHA_THOUSAND"]].values:
        if np.isnan(shipping_price) or shipping_price < 1e6:
            if np.isnan(nc_grade_price) or nc_grade_price < 1e6:
                if np.isnan(new_car_price) or new_car_price < 1e6:
                    if np.isnan(exha_dict[exha_thousand]) or exha_dict[exha_thousand] < 1e6:
                        val = global_mean
                    else:
                        val = exha_dict[exha_thousand]
                else:
                    val = new_car_price
                price_list.append(val)
            else:
                price_list.append(nc_grade_price)
        else:
            price_list.append(shipping_price) 

    return_df['PRICE'] = price_list
    return return_df

df_2 = preprocessing_PRICE(df_1)
```

```python
####
def preprocessing_year(df):
    return_df = df.copy()
    return_df['YEAR_adj'] = return_df.apply(lambda x: 0 if np.isnan(x['SUCCYMD']) or np.isnan(x['YEAR']) else int(str(x['SUCCYMD'])[:4])-x['YEAR'] , axis=1)
    #평균 주행거리 TRA_YEAR
    return_df['TRA_YEAR'] = return_df.apply(lambda x: 0 if np.isnan(x['TRAVDIST']) or x['YEAR_adj']==0  else x['TRAVDIST']/x['YEAR_adj'], axis=1)
    TRAV_AVG = return_df['TRA_YEAR'].mean()
    return_df['TRAVDIST'] = return_df.apply(lambda x: TRAV_AVG*x['YEAR_adj'] if np.isnan(x['TRAVDIST']) else x['TRAVDIST'], axis=1)
    return return_df

df_3 = preprocessing_year(df_2)
```

```python
#####
# string -> int
def preprocessing_string(df):
    return_df = df.copy()
    string_dict = {}
    string_col_list = []
    for col in df.columns:
        if df[col].dtypes == object:
            string_col_list.append(col)
    for col in string_col_list:
        string_dict[col] = {}
        count = 0
        for val in df[col].values:
            if val not in string_dict[col]:
                count += 1
                string_dict[col][val] = count
    for col in string_col_list:
        return_df[col] = return_df[col].apply(lambda x: string_dict[col][x])
    return return_df, string_col_list

df_4, string_col_list = preprocessing_string(df_3)
####
```

```python
#####
def preprocessing(df):
    df = preprocessing_EXHA(df)
    df = preprocessing_PRICE(df)
    df = preprocessing_year(df)
    df, string_col_list = preprocessing_string(df)
    return df, string_col_list

#####
```

```python
def make_train_val(df, use_x_col_list):
    def splitData(df, ratio, y_column):        
        columns_ = df.columns
        # Subsample the data
        mask = list(range(0,df.shape[0], ratio))
        X_val = df.iloc[mask, :].drop(y_column, 1)
        y_val = df.iloc[mask][y_column]
        
        mask = ~df.index.isin(mask)
        X_train = df.loc[mask, :].drop(y_column, 1)
        y_train = df.loc[mask, y_column]
        
        return X_train, y_train, X_val, y_val

    def z_normalize(_df, col_list):
        df = _df.copy()
        cache_dict = {}
        for col in col_list:
            cache_dict[col] = {"mean": df[col].mean(), "std": df[col].std()}
            if cache_dict[col]["std"] == 0:
                cache_dict[col]["std"] = 1
            df[col] = (df[col] - cache_dict[col]["mean"])/cache_dict[col]["std"]
        return df, cache_dict

    def z_normalize_val(_df, col_list, cache_dict):
        df = _df.copy()
        for col in col_list:
            df[col] = (df[col] - cache_dict[col]["mean"])/cache_dict[col]["std"]
        return df

    preprocessed_df = df.copy()
    X_train, y_train, X_val, y_val = splitData(preprocessed_df, 5, y_column = 'SUCCPRIC') 
    X_train_norm, cache = z_normalize(X_train, use_x_col_list)
    X_val_norm=z_normalize_val(X_val, use_x_col_list, cache)
    print("maked train, validate data set")

    return X_train_norm, y_train, X_val_norm, y_val

def make_best_rf(X_train_norm, y_train, X_val_norm, y_val):
    best_rf = None  
    best_mse = 1e20
    best_hyperparameter = None
    results = {}
    n_estimators = [5, 50, 100]
    max_depths = [5, 50, 100]
    ects = None

    for max_depth in max_depths:
        for  n_estimator in n_estimators:
            regressor = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator, random_state=0)
            rf = regressor.fit(X_train_norm, y_train)
            y_pred = rf.predict(X_val_norm)
            mse = metrics.mean_squared_error(y_val, y_pred)
            results[(max_depth, n_estimator)] = mse
            print("max_depth: {} n_estimator: {} mse: {}".format(max_depth, n_estimator, mse))
            if mse < best_mse:
                best_mse = mse
                best_rf = rf
                best_hyperparameter = {"max_depth": max_depth, "n_estimator": n_estimator}
    print(best_hyperparameter)
    return best_rf


####
```

```python
def plot_feature_importance(importance_, features_,model_type):
    dict_ = {'feature importance' : importance_, 'features' : features_}
    df = pd.DataFrame(dict_)
    df.sort_values(by=['feature importance'], ascending=False,inplace=True)
    plt.figure(figsize=(10,10))
    sns.barplot(x=df['feature importance'], y=df['features'])
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

#plot_feature_importance(regressor.feature_importances_, X_val.columns, 'RANDOM FOREST ')

def print_feature_importance(best_rf, X_val_norm):
    imp=[]
    for importance, feature in zip(best_rf.feature_importances_, X_val_norm.columns):
        imp.append((importance, feature))
    imp_sorted=sorted(imp)
    for importance, feature in imp_sorted:
        print(importance, feature)
####
```

```python
# coninuos variable
"""
배기량, 가격 3종류 / 연식, 낙찰일자 / 주행거리/
"""
pd.set_option('display.max_rows', 200)

preprocessed_df = df_4.copy()
cor_df = preprocessed_df.corr()['SUCCPRIC']
cor_df = cor_df.dropna()
cor_df_abs = cor_df.abs()
print(cor_df_abs.sort_values()[-11:-1])

#####
```

```python
#-------------------------연속형변수만 랜덤포레스트
train_y_list = ["SUCCPRIC"]
numeric_col_list = []
temp_string_col_dict = {}
for col in string_col_list:
    temp_string_col_dict[col] = True

for col in preprocessed_df.columns:
    if col not in temp_string_col_dict:
        if col != "SUCCPRIC":
            numeric_col_list.append(col)
    
print(numeric_col_list)
numeric_preprocessed_df = preprocessed_df[numeric_col_list+train_y_list] 
temp_numeric_df = numeric_preprocessed_df.copy().dropna()
X_train_norm, y_train, X_val_norm, y_val = make_train_val(temp_numeric_df, numeric_col_list)
start = time()
best_rf = make_best_rf(X_train_norm, y_train, X_val_norm, y_val)
print("time: ", time()-start, "s")

print_feature_importance(best_rf, X_val_norm)

#####
```

```python
# string variable select: RF important 계산
"""
연료명, 차량명, 색, 용도
"""
for col in string_col_list:
    print(col, len(df_3[col].value_counts()), df_3.shape[0])

#####
```

```python
string_preprocessed_df = preprocessed_df[string_col_list+train_y_list] 
temp_string_df = string_preprocessed_df.copy().dropna()
X_train_norm, y_train, X_val_norm, y_val = make_train_val(temp_string_df, string_col_list)
start = time()
best_rf = make_best_rf(X_train_norm, y_train, X_val_norm, y_val)
print("time: ", time()-start, "s")

print_feature_importance(best_rf, X_val_norm)
```

```python
# 최종 select된 변수로 RF 실행
use_numeric_list = ["PRICE", "EXHA_ADJ","TRAVDIST", "YEAR_adj"]
use_string_list = ["CARNM", "COLOR", "USEUSENM", "FUELNM"]
use_x_list = use_numeric_list + use_string_list

final_preprocessed_df = preprocessed_df[use_x_list+train_y_list]
X_train_norm, y_train, X_val_norm, y_val = make_train_val(final_preprocessed_df, use_x_list)
start = time()
best_rf = make_best_rf(X_train_norm, y_train, X_val_norm, y_val)
print("time: ", time()-start, "s")

#####
```

```python
def mape(y_pred, y_true ): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def rmse(y_pred, y_true):
    return np.sqrt(np.mean(np.square(y_pred- y_true)))

y_train_pred = best_rf.predict(X_train_norm)
print('Train RMSE: ', rmse(y_train_pred, y_train))
print('Train MAPE: ', mape(y_train_pred, y_train))
y_val_pred = best_rf.predict(X_val_norm)
print('Validation RMSE: ', rmse(y_val_pred, y_val))
print('Validation MAPE: ', mape(y_val_pred, y_val))

y_val_pred=best_rf.predict(X_val_norm)
plt.scatter(y_val, y_val_pred, alpha=0.4)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Tree Regression')
plt.show()
```

> ## P1-3. System Implementation
>
> - df_test에 기반하여 실제 사용을 위한 system implemetation 작업 진행

```python
# test 처리
# df_test 에서 true y를 제거후에도 전처리가 잘 작동하는지 확인해봄
df_test_drop = df_test.drop(["SUCCPRIC"], axis=1)
preprocesed_df_test, string_col_list = preprocessing(df_test_drop)
use_numeric_list = ["PRICE", "EXHA_ADJ","TRAVDIST", "YEAR_adj"]
use_string_list = ["CARNM", "COLOR", "USEUSENM", "FUELNM"]
use_x_list = use_numeric_list + use_string_list
input_x = preprocesed_df_test[use_x_list]

pred_y = best_rf.predict(input_x)
for idx, pred_val in enumerate(pred_y):
  print("{} 번 중고차 시세 예측 가격은: {}입니다".format(idx, pred_val))
```

