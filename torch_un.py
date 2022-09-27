import  matplotlib as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import train_test_split  # train, valid set 제작
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV  # 파라미터 설정 고민을 줄여주는 고마운 친구
from sklearn.metrics import make_scorer  # loss function 커스터마이징

# 시각화 설정
sns.set_context("talk")
sns.set_style("white")
font_title = {"color":"gray"}

# Linux 한글 사용 설정
plt.rcParams['font.family']=['NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 펭귄 데이터셋 불러오기
train=pd.read_csv("/home/a/PycharmProjects/train.csv",encoding='euc-kr')
test=pd.read_csv("/home/a/PycharmProjects/test.csv",encoding='euc-kr')
submission=pd.read_csv('/home/a/PycharmProjects/sample_submission.csv', encoding='euc-kr')

train[['num', '비전기냉방설비운영','태양광보유']]
ice={}
hot={}
count=0
for i in range(0, len(train), len(train)//60):
    count +=1
    ice[count]=train.loc[i,'비전기냉방설비운영']
    hot[count]=train.loc[i,'태양광보유']

for i in range(len(test)):
    test.loc[i, '비전기냉방설비운영']=ice[test['num'][i]]
    test.loc[i, '태양광보유']=hot[test['num'][i]]




# print(test.head(10))
# print(train.head(10))
# 학습용set 생성
train.drop('date_time', axis=1, inplace=True)  # 학습에 불필요한 날짜 제거
train_x=train.drop('전력사용량(kWh)', axis=1)  # 문제
train_y=train[['전력사용량(kWh)']]  # 정답

X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=1221)
# encoder
from sklearn.preprocessing import RobustScaler

# machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def get_model_0(X_cols, degree=1, method="lr"):
    X_cols_ = deepcopy(X_cols)

    #     # 1-1.categorical feature에 one-hot encoding 적용
    #     cat_features = list(set(X_cols) & set(["species", "island", "sex"]))
    #     cat_transformer = OneHotEncoder(sparse=False, handle_unknown="ignore")

    # 1-2.numerical feature는 Power Transform과 Scaler를 거침
    num_features = list(set(X_cols))
    num_features.sort()
    num_transformer = Pipeline(steps=[("polynomial", PolynomialFeatures(degree=degree)),
                                      ("scaler", RobustScaler())
                                      ])

    # 1. 인자 종류별 전처리 적용
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    # 2. 전처리 후 머신러닝 모델 적용
    if method == "lr":
        ml = LinearRegression(fit_intercept=True)
    elif method == "rf":
        ml = RandomForestRegressor()

    # 3. Pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("ml", ml)])

    return model
from sklearn import set_config
set_config(display='diagram')
model_0 = get_model_0(list(X_train.columns), degree=1, method="lr")
model_0

X_train_pp = model_0["preprocessor"].fit_transform(X_train)
print(X_train_pp.shape)

model_0.fit(X_train, y_train)

model_1 = get_model_0(list(X_train.columns), degree=1, method="rf")
model_1.fit(X_train, y_train.values.ravel())
model_1

from torch import optim
from torch.optim.lr_scheduler import CyclicLR

import torch
import torch.nn as nn


class RegressorModule(nn.Module):
    def __init__(self, ninput=8, init_weights=True):
        super(RegressorModule, self).__init__()

        self.model = nn.Sequential(nn.Linear(ninput, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 12),
                                   nn.ReLU(),
                                   nn.Linear(12, 8),
                                   nn.ReLU(),
                                   nn.Linear(8, 1),
                                   )
        if init_weights:
            self._initialize_weights()

    def forward(self, X, **kwargs):
        return self.model(X)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

X_train_tensor = torch.Tensor(pd.get_dummies(X_train).astype(np.float32).values)
y_train_tensor = torch.Tensor(y_train.astype(np.float32).values)

net = RegressorModule()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


loss_func = RMSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

losses = []
for i in range(400):
    optimizer.zero_grad()
    output = net.forward(X_train_tensor)
    loss = loss_func(output, y_train_tensor.view(-1, 1))
    loss.backward()
    optimizer.step()

    losses.append(loss)

fi_los = [fl.item() for fl in losses ]
plt.plot(range(400), fi_los)
plt.ylabel('RMSE Loss')
plt.xlabel('Epoch')
plt.show()

# numpy array를 pytorch tensor로 변환
X_test_tensor = torch.Tensor(pd.get_dummies(X_valid).astype(np.float32).values)

# 예측값
y_pred_train_tensor = net.forward(X_train_tensor)
y_pred_test_tensor = net.forward(X_test_tensor)

# pytorch tensor를 다시 numpy array로 변환
y_pred_train = y_pred_train_tensor.detach().numpy()
y_pred_test = y_pred_test_tensor.detach().numpy()

from skorch import NeuralNetRegressor
from sklearn.base import BaseEstimator, TransformerMixin


def get_model_T(X_cols, degree=1, method="lr"):
    X_cols_ = deepcopy(X_cols)


    cat_features = list(set(X_cols) & set(["비전기냉방설비운영","태양광보유",'dum']))
    cat_transformer = OneHotEncoder(sparse=False, handle_unknown="ignore")

    # 1-2.numerical feature는 Power Transform과 Scaler를 거침
    num_features = list(set(X_cols))
    num_features.sort()
    num_transformer = Pipeline(steps=[("polynomial", PolynomialFeatures(degree=degree)),
                                      ("scaler", RobustScaler())
                                      ])

    # 1. 인자 종류별 전처리 적용
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])


    # 2. float64를 float32로 변환
    class FloatTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, x):
            return np.array(x, dtype=np.float32)

    # 3. 전처리 후 머신러닝 모델 적용
    if method == "lr":
        ml = LinearRegression(fit_intercept=True)
    elif method == "rf":
        ml = RandomForestRegressor()
    elif method == "torch":
        ninput = len(num_features)
        if "비전기냉방설비운영" in cat_features:
            ninput += 3
        if "테양광보유" in cat_features:
            ninput += 3
        if "dum" in cat_features:
            ninput += 2



        net = NeuralNetRegressor(RegressorModule( init_weights=False),
                                 max_epochs=1000, verbose=0,
                                 warm_start=True,
                                 #                          device='cuda',
                                 criterion=RMSELoss,
                                 optimizer=optim.Adam,
                                 optimizer__lr=0.01
                                 )
        ml = net

    # 3. Pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("float64to32", FloatTransformer()),
                            ("ml", ml)])

    return model

model_T = get_model_T(list(X_train.columns), degree=1, method="torch")
model_T.fit(X_train, y_train.astype(np.float32).values.reshape(-1, 1))

#

from sklearn.inspection import permutation_importance

# Linear Regression
pi_0 = permutation_importance(model_0, X_valid, y_valid, n_repeats=30, random_state=0)

# Random Forest
pi_1 = permutation_importance(model_1, X_valid, y_valid, n_repeats=30, random_state=0)

# Neural Network
pi_T = permutation_importance(model_T, X_valid, y_valid, n_repeats=30, random_state=0)

# 시각화
fig, axs = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True, sharey=True)

for ax, pi, title in zip(axs, [pi_0, pi_1, pi_T], ["Linear Reg.", "Random Forest", "Neural Net"]):
    ax.barh(X_valid.columns, pi.importances_mean, xerr=pi.importances_std, color="orange")
    ax.invert_yaxis()
    ax.set_xlim(0, )
    ax.set_title(title, fontdict=font_title, pad=16)
