# P2P 대출평가(EFSVM)

> ## 1. Environmental Set-up & Data Loading

```python
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# enter the foldername in your Google Drive where you have saved the unzipped
FOLDERNAME =  'ADX/'

assert FOLDERNAME is not None, 'ERROR'

%cd drive/My\ Drive
%cp -r $FOLDERNAME ../../
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import spmatrix, sparse
from cvxopt import solvers as cvxopt_solvers
from time import time

from scipy.stats import entropy
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import psutil
import gc
```

```python
# 모델 수립을 위한 Train/Validation Set
df = pd.read_csv('./ADX/Data_P2P/P2_dataset.csv', encoding ='cp949') 

# 시스템 구현을 위한 Test Set 샘플
df_test = pd.read_csv('./ADX/Data_P2P/P2_dataset_test_sample.csv', encoding ='cp949') 

print("남은 메모리 (%)", psutil.virtual_memory().available*100/psutil.virtual_memory().total)
```

> ## 2. Modelling
>
> - df를 사용하여 예측모형 수립
> - Feature engineering에 대한 토의/구현 진행
> - Computation time을 고려하여 전체 데이터를 완전히 사용하지 말고 Sampling하여 Model Train/Vadliation을 진행하는 것을 추천
> - Sampling을 단 한번 한 것으로 모형 Train 한 것이 과연 옳은 것인지에 대해 팀원들과 고민해볼 것. 해결 방법은 없을지 서치해보는 것도 하나의 task임.

```python
def split_x_y(df):
    """
    df: 원본 dataset
    return x, y
    """
    y_name = "loan_status"
    pos_name = "Charged Off"
    neg_name = "Fully Paid"

    df[y_name] = df[y_name].replace(pos_name, 1)
    df[y_name] = df[y_name].replace(neg_name, -1)
    y = df[y_name]
    X = df.drop([y_name], axis=1)

    return X, y
```

```python
def preprocessing(ret_df_x, predict_use_feature, preprocessing_use_feature):
    ####
    # 문자열 변수 -> 오더링(y=-1일 가능성이 높은 카테고리에 높은 숫자 부여) & 결측값 처리
    grade = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1, np.nan:1}
    sub_grade = {'1':.8, '2':.6, '3':.4, '4':.2, '5':0, np.nan:0}
    ownership = {'MORTGAGE':4 ,'OWN':3, 'RENT':2, 'NONE':1, 'ANY':1, 'OTHER':1, np.nan:1}

    ret_df_x['grade'] = ret_df_x['grade'].apply(lambda x: grade[x])
    ret_df_x['sub_grade'] = ret_df_x['sub_grade'].apply(lambda x: sub_grade[x[1]])
    ret_df_x['new_grade'] = ret_df_x['grade']+ret_df_x['sub_grade']
    ret_df_x['home_ownership'] = ret_df_x['home_ownership'].apply(lambda x: ownership[x])    
    ####
    # 결측값 처리
    #ret_df_x['last_pymnt_amnt'] = ret_df_x['last_pymnt_amnt'].fillna(0)
    ret_df_x['inq_last_6mths'] = ret_df_x['inq_last_6mths'].fillna(0)
    ret_df_x['installment'] = ret_df_x['installment'].fillna(0)
    ##
    """
    annual_inc
      - 결측값 존재시, new_grade기준으로 판단
      - 만약 new_grade가 결측이면 평균값으로 대체
    """
    annual_inc_mean_dict = dict(ret_df_x[["annual_inc", "new_grade"]].groupby(['new_grade']).mean()['annual_inc'])
    val_list = []
    for annual_inc, new_grade in ret_df_x[['annual_inc', 'new_grade']].values:
        if np.isnan(annual_inc):
            val_list.append(annual_inc_mean_dict[new_grade])
        else:
            val_list.append(annual_inc)
    ret_df_x['annual_inc'] = val_list
    ##
    """
    int_rate
      - 결측값 존재시, new_grade기준으로 판단
      - 만약 new_grade가 결측이면 평균값으로 대체
    """
    int_rate_mean_dict = dict(ret_df_x[["int_rate", "new_grade"]].groupby(['new_grade']).mean()['int_rate'])
    val_list = []
    for int_rate, new_grade in ret_df_x[['int_rate', 'new_grade']].values:
        if np.isnan(annual_inc):
            val_list.append(int_rate_mean_dict[new_grade])
        else:
            val_list.append(int_rate)
    ret_df_x['int_rate'] = val_list
    ##
    val_list = []
    for dti, istallment, annual_inc in zip(ret_df_x['dti'].values, ret_df_x['installment'].values, ret_df_x['annual_inc'].values):
        if np.isnan(dti):
            val_list.append(istallment/(annual_inc/12))
        else:
            val_list.append(dti)
    ret_df_x['dti'] = val_list
    return ret_df_x[predict_use_feature]
```

```python
predict_use_feature = ["home_ownership",  "int_rate", "dti", "annual_inc", "inq_last_6mths"] #"last_pymnt_amnt",
preprocessing_use_feature = ["grade", "sub_grade", "installment"] + predict_use_feature
x_train, y_train = split_x_y(df)
x_train = preprocessing(x_train, predict_use_feature, preprocessing_use_feature)
```

```python
lst = [df]
del df
del lst
gc.collect()
```

```python
def extract_sample(x_train, y_train, test_size=0.9,random_state=0):
    print("train_size: ", 1-test_size, "random_state: ", random_state)
    small_x_train, _, small_y_train, _ = train_test_split(x_train, y_train, test_size=test_size, stratify=y_train, random_state=random_state)  
    return small_x_train, small_y_train
```

```python
def make_si(X_, y_, k, m, beta):
    X = X_.copy()
    y = y_.copy()
    start = time()
    pos_val = 1
    neg_val = -1
    Ypos = y[y==pos_val].index
    Yneg = y[y==neg_val].index
    Entropy_Yneg = pd.DataFrame()
    distNeg = pd.DataFrame(distance.cdist(X, X.loc[Yneg], "euclidean"), index = X.index, columns=Yneg)
    ################
    for i in Yneg:
        numP = np.sum(y.loc[distNeg.loc[:, i].sort_values()[1:k+1].index] == pos_val)
        numN = k - numP
        probP = numP/k
        probN = numN/k
        H = entropy([probP, probN])
        Entropy_Yneg.loc[i, "numP"] = numP
        Entropy_Yneg.loc[i, "numN"] = numN
        Entropy_Yneg.loc[i, "probP"] = probP
        Entropy_Yneg.loc[i, "probN"] = probN
        Entropy_Yneg.loc[i, "H"] = H


    print("before del distNeg 남은 메모리 (%)", psutil.virtual_memory().available*100/psutil.virtual_memory().total)
    lst = [distNeg]
    del distNeg
    del lst
    gc.collect()
    print("after del distNeg 남은 메모리 (%)", psutil.virtual_memory().available*100/psutil.virtual_memory().total)
    #################
    Emax, Emin = Entropy_Yneg['H'].max(), Entropy_Yneg['H'].min()
    FM = {}
    for l in range(1, m+1):
        thrUp = Emin + l/m*(Emax-Emin)
        thrLow = Emin + (l-1)/m*(Emax-Emin)
        if m == l:
            Entropy_Yneg.loc[(Entropy_Yneg['H'] >= thrLow) & (Entropy_Yneg['H'] <= thrUp), 'subi'] = l
            Entropy_Yneg.loc[(Entropy_Yneg['H'] >= thrLow) & (Entropy_Yneg['H'] <= thrUp), "FM"] = 1-beta*(l-1)
        elif m != l:
            Entropy_Yneg.loc[(Entropy_Yneg['H'] >= thrLow) & (Entropy_Yneg['H'] < thrUp), 'subi'] = l
            Entropy_Yneg.loc[(Entropy_Yneg['H'] >= thrLow) & (Entropy_Yneg['H'] < thrUp), "FM"] = 1-beta*(l-1)
    si = pd.DataFrame(index=X.index)
    si.loc[Ypos, "si"] = 1
    si.loc[Entropy_Yneg.index, "si"] = Entropy_Yneg["FM"].values


    print("before del Entropy_Yneg 남은 메모리 (%)", psutil.virtual_memory().available*100/psutil.virtual_memory().total)
    lst = [Entropy_Yneg]
    del Entropy_Yneg
    del lst
    gc.collect()
    print("after del Entropy_Yneg 남은 메모리 (%)", psutil.virtual_memory().available*100/psutil.virtual_memory().total)

    si = np.array(si['si'])
    print("make si done {}s".format(time()-start))
    return si
```

```python
def Kernel_(x1, x2, params = 0, type_ = 'default') :
    """
    x1: (N , D)
    x2: (B , D)
    """
    if type_ == 'rbf' :
        """
        using broadcasting
        (N, B) = (N, 1) + (1, B) - (N, B)
        """
        Kernel = np.exp(- (np.sum(x1 **2, axis = 1).reshape(-1,1) + np.sum(x2 **2, axis = 1).reshape(1,-1) - 2 * x1 @ x2.T)* params)
        return Kernel
    elif type_ == 'default' :
        Kernel = np.dot(x1, x2.T)
        return Kernel
def scailing(X, type_= "standard" ,cache={}):
    if type_ == "standard":
        if len(cache) == 0:
            print("make cache, type: "+type_)
            cache["mean"] = X.mean(axis=0)
            cache["std"] = X.std(axis=0)
        elif len(cache) != 0:
            print("using cache, type: "+type_)
        scaling_X = (X-cache["mean"])/cache["std"]
        print("cahce ", cache)
        return scaling_X, cache
def find_optimal(X, y, si, C, Gamma, step=0, initvals=None):
    start = time()
    num_x, dim_x = X.shape
    H = Kernel_(X, X, type_="rbf", params=Gamma)*1.
    H *= y@y.T 
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((num_x, 1)))
    G = spmatrix([-1 for _ in range(num_x)]+[1 for _ in range(num_x)], [i for i in range(num_x)]+[num_x+i for i in range(num_x)], [i for i in range(num_x)]+[i for i in range(num_x)])
    #G = cvxopt_matrix(np.vstack((-np.eye(num_x),np.eye(num_x))))
    h = cvxopt_matrix(np.hstack((np.zeros(num_x), np.array(si) * C)))
    #h = spmatrix(np.array(si)*C, [num_x+i for i in range(num_x)], [0 for i in range(num_x)])
    #print(isinstance(h, cvxopt_matrix), h.typecode, h.size)
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    if step == 0:
      sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    else:
      sol = cvxopt_solvers.qp(P, q, G, h, A, b, list(initvals))
    alphas = np.array(sol['x'])
    print("make optimal sol done time: {}s".format(time()-start))
    return alphas
def calc_S_b(X, y, alphas, type_="rbf", Gamma=2):
    """
    X: (N, D)
    y: (N, 1)
    alphas: (N, 1)
    
    return
        suprt_vector_index: (S, 1), List(boolean)
        bias: 
              mean{(S, 1) - sum(K((N, D), (S, D))*(N, 1)*(N, 1), axis=0).reshape(-1, 1)}
            = mean{(S, 1) - sum((N, S)*(N, 1)*(N, 1), axis=0).reshape(-1, 1)}
            = mean{(S, 1) - sum((N, S), axis=0).reshape(-1, 1)}
            = mean{(S, 1) - (S, 1)}
            = (1, 1)
    """
    if type_ == 'rbf':
        suport_vector_index = ((alphas > 1e-4) & (alphas < C-1e-4)).flatten()
        bias = np.mean(y[suport_vector_index] - np.sum(Kernel_(X, X[suport_vector_index]  , params = Gamma, type_ = type_)* y * alphas , axis = 0).reshape(-1,1))
    return suport_vector_index, bias
```

```python
# Hyper parameter
C = 10
Gamma = 2 
k = 7 # Number of Nearest Neighbor
m = 10
beta = 1/18
```

```python
# 0.94 성공
test_size = 0.95

step = 0
num_batch = 1
print(num_batch)
for random_state in range(num_batch):
  small_x_train, small_y_train = extract_sample(x_train, y_train, test_size=test_size, random_state = random_state)
  print("\n===train: {}=====".format(random_state))
  # print("num of train", small_x_train.shape[0])
  # numP = len(small_y_train[small_y_train==1])
  # numN = len(small_y_train[small_y_train==-1])
  # print("y == 1", numP/(numP+numN))
  # print("y == -1", numN/(numP+numN))
  # print("==============")
  si_train = make_si(small_x_train, small_y_train, k, m, beta)
  train_x, cache = scailing(np.array(small_x_train), type_="standard", cache={})
  train_y = np.array(small_y_train, dtype=np.float64).reshape(-1, 1)
  if step == 0:
    alphas = find_optimal(train_x, train_y, si_train, C, Gamma)
    #past_alphas = alphas
  else:
    alphas = find_optimal(train_x, train_y, si_train, C, Gamma, step, initvals=alphas)
  #past_alphas = alphas
  step += 1
  
  lst = [small_x_train, small_y_train]
  del small_x_train
  del small_y_train
  gc.collect()
```

```python
Suport_vector_index_train, bias_train = calc_S_b(train_x, train_y, alphas, type_="rbf", Gamma=Gamma)
```

```python
def pred(train_x, train_y, test_x, aplahs, bias, type_="rbf", Gamma=2):
    """
    N: number of train_x
    D: dim of x
    B: number of test_x
    
    train_x: (N, D)
    train_y: (N, D)
    test_x: (B, D)
    alpahs: (N, 1)
    bias: (1)
    
    pred_sol:
        (B, 1) 
            = sum(K((N, D), (B, D))*(N, 1)*(N, 1), axis=0).reshape(-1, 1) + (1)
            = sum((N, B)*(N, 1)*(N, 1), axis=0).reshape(-1, 1) + (1)
            = sum((N, B), axis=0).reshape(-1, 1) + (1)
            = (1, B).reshape(-1, 1) + (1)
            = (B, 1) + (1)
            = (B, 1)
    """
    if type_ == "rbf":
        pred_sol = np.sign(np.sum(Kernel_(train_x, test_x, params=Gamma, type_=type_)*train_y*alphas, axis=0).reshape(-1, 1) +  bias)
        return pred_sol
```

```python
def Convolution(pred, real) :
    pred = np.array(pred)
    y = np.array(real)
    TP = np.sum((pred == 1) & (y == 1))
    FP = np.sum((pred == 1) & (y != 1))
    FN = np.sum((pred != 1) & (y == 1))
    TN = np.sum((pred != 1) & (y != 1))
    return TP, FP, FN, TN

def accuracy(TP, FP, FN, TN):
    return (TP+TN)/(TP+FP+FN+TN)

def acc_precision_recall(X) :
    TP,FP,FN,TN = X
    eps = 1e-10
    return (TP + TN) / (TP + FP + FN + TN+eps), TP / (TP + FP+eps), TP / (TP + FN+eps)
```

```python
pred_sol_train = pred(train_x, train_y, train_x, alphas, bias_train, type_="rbf", Gamma=Gamma)
print(acc_precision_recall(Convolution(pred_sol_train, train_y)))
```

> ## 3. System Implementation
>
> - df_test에 기반하여 실제 사용을 위한 system implemetation 작업 진행
> - Performance에 대한 평가데이터는 df_test로 지난 프로젝트와 마찬가지로 Data Pre-processing이 System implementation에 동시 구현
> - 특히, scaler를 사용할 경우 cache를 반드시 사용하여 올바른 system implementation이 되도록 할 것!!! (주의!!!)

```python
x_test, y_test = split_x_y(df_test)
preprocessing_x_test = preprocessing(x_test, predict_use_feature, preprocessing_use_feature)
test_x, cache = scailing(np.array(preprocessing_x_test), type_="standard", cache=cache)
test_y = np.array(y_test, dtype=np.float64).reshape(-1, 1)
pred_sol_test = pred(train_x, train_y, test_x, alphas, bias_train, type_="rbf", Gamma=Gamma)
print((1-test_size)*100, "%")
print("batch num", num_batch)
print(acc_precision_recall(Convolution(pred_sol_test, test_y)))
```

```python
for i in range(3):
    print("input x: ")
    print(x_test.iloc[i])
    print()
    pred = pred_sol_test[i][0]
    if pred == 1:
        print({"Prediction": "Charged Off"})
    elif pred == -1:
        print({"Prediction: Fully Paid"})
    print("="*100)
    print()
```

