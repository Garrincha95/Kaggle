### [Kaggle 'Titanic: Machine Learning from Disaster'](https://www.kaggle.com/c/titanic)

## 목차

**[1. 프로젝트 소개](#1-프로젝트-소개)**

**[2. 코드 리뷰](#2-코드-리뷰)**

## 1. 프로젝트 소개

> **개요**: Kaggle 'Titanic: Machine Learning from Disaster'의 [Dataset](https://www.kaggle.com/c/titanic/data)을 사용하고 Python을 사용하여 분석 및 생존 예측

> **스택**: Python

> **툴**: Jupyter notebook

> **설치 방법**
>
> * Repository 다운로드 또는 git clone
> * 해당 명령어로 requirements.txt 설치
> * ``` pip install -r requirements.txt```
> * Jupyter notebook 실행 및 Titanic.ipynb 클릭

> **디펜던시**: Python, Numpy, Pandas, SciKit-Learn, Matplotlib, Seaborn, Missingno, Re, Warnings, Os, Keras

---

## 2. 코드 리뷰

1. **Import**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2.5)

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

---

2. **데이터 읽기**

```python
train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')

train_df.describe()

test_df.describe()

train_df.isnull().sum()

test_df.isnull().sum()

train_df.isnull().sum()/len(train_df)

test_df.isnull().sum()/len(test_df)

import missingno as msno
msno.bar(df = train_df)

msno.bar(df = test_df)
```

---

3. **생존과 연관 된 컬럼 찾기**

```python
# Pclass 컬럼: Pclass가 좋을수록 생존률 높음

train_df.Survived == 1

train_df[train_df.Survived == 1].Pclass.value_counts()

pd.crosstab(train_df.Pclass, train_df.Survived, margins=True)

# Sex 컬럼: 여성이 생존 확률이 높음

pd.crosstab(train_df.Sex, train_df.Survived, margins=True)

sns.factorplot('Pclass', 'Survived', hue='Sex', data = train_df)

# Age 컬럼: 나이 어린 사람이 생존 확률이 높음

surv_age = train_df.Age[train_df.Survived == 1]

dead_age = train_df.Age[train_df.Survived == 0]

plt.figure(figsize=(10,10))
surv_age.plot(kind = 'kde')
dead_age.plot(kind = 'kde')
plt.legend(['surv_age','dead_age'])

# Embarked 컬럼: 비슷한 생존률, C가 높은 것은 Pclass가 높은 사람이 많이 타서 높음

pd.crosstab(train_df.Embarked, train_df.Survived, margins=True)

pd.crosstab(train_df.Embarked, train_df.Pclass, margins=True)

# 상관관계 & 기울임 정보

train_df.corr().Survived

train_df.skew()

test_df.skew()

plt.hist(train_df.Fare)

plt.hist(test_df.Fare)

train_df.Fare = np.log1p(train_df.Fare)
test_df.Fare = np.log1p(test_df.Fare)

plt.hist(train_df.Fare)

plt.hist(test_df.Fare)
```

---

4. **전처리**

```python
# 가족 수 컬럼 추가: 4인 가족 생존률 높음, 5~11인 가족 생존률 낮음

train_df['Family'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family'] = test_df['SibSp'] + test_df['Parch'] + 1

pd.crosstab(train_df.Family, train_df.Survived, margins=True)

# Sir 컬럼 추가

train_name_list = train_df.Name.tolist()
test_name_list = test_df.Name.tolist()

import re
sir_list = re.compile('Mrs|Mr|Miss|Master|Don|Dr|Ms|Major|Mlle|Mne|Countess|Lady|Don|Countess')

sir = []
def transfer(name_list):
    for name in name_list:
        if sir_list.search(name) != None:
            if sir_list.search(name)[0] in ['Countess','Lady','Mrs']:
                sir.append('Mrs')
            elif sir_list.search(name)[0] in ['Mme', 'Ms', 'Mlle']:
                sir.append('Miss')
            elif sir_list.search(name)[0] in ['Don','Dr','Major']:
                sir.append('Mr')
            else:
                sir.append(sir_list.search(name)[0])
        else:
            sir.append('Other')
            
transfer(train_name_list)

pd.Series(sir)

pd.Series(sir).value_counts()

train_df['Sir'] = pd.Series(sir)

pd.crosstab(train_df.Sir, train_df.Survived, margins=True)

transfer(test_name_list)

pd.Series(sir)

pd.Series(sir).value_counts()

test_df['Sir'] = pd.Series(sir)

test_df.head()

# 결측치 처리 (Age, Embarked, Fare)
# 나이 결측치는 Sir 컬럼의 값을 참조해 같은 Sir 값을 가진 행의 평균 값으로 대체


train_sir = train_df.groupby('Sir').mean()

test_sir = test_df.groupby('Sir').mean()

train_sir.Age

test_sir.Age

train_age_nan_index = train_df.index[train_df.Age.isnull()]
test_age_nan_index = test_df.index[test_df.Age.isnull()]

train_age_nan_index

test_age_nan_index

def process_age_nan(age_nan_index):
    if len(age_nan_index) == len(train_age_nan_index):
        for index in age_nan_index:
            row_sir = train_df.loc[index]['Sir']
            print('index:',index, 'row_sir:', row_sir)
            sir_mean = train_sir.loc[row_sir]['Age']
            print('sir_mean:', sir_mean)
            train_df.at[index, 'Age'] = sir_mean
    else:
        for index in age_nan_index:
            row_sir = test_df.loc[index]['Sir']
            print('index:',index, 'row_sir:', row_sir)
            sir_mean = test_sir.loc[row_sir]['Age']
            print('sir_mean:', sir_mean)
            test_df.at[index, 'Age'] = sir_mean
            
process_age_nan(train_age_nan_index)

process_age_nan(test_age_nan_index)

train_df.Age.isnull().sum()

test_df.Age.isnull().sum()

# 승선(승선한 항구)값 결측치는 Pcalss 칼럼의 값을 참조해 가장 많은 값을 가진 행의 대체

train_emb_nan_index = train_df.index[train_df.Embarked.isnull()]

train_df.Embarked.fillna('S', inplace=True)
train_df.Embarked.isnull().sum()

# 표 값 결측치는 Pclass 칼럼의 값을 참조해 같은 Pclass 값을 가진 행의 평균값으로 대체

test_fare = test_df.groupby('Pclass').mean()

test_fare_nan_index = test_df.index[test_df.Fare.isnull()]

for index in test_fare_nan_index:
    row_pclass = test_df.loc[index]['Pclass']
    print('index:',index, 'row_pclass:', row_pclass)
    fare_mean = test_fare.loc[row_pclass]['Fare']
    print('fare_mean:', fare_mean)
    test_df.at[index, 'Fare'] = fare_mean
    
test_df.Fare.isnull().sum()
```

---

5. **범주형 데이터 처리**

```python
# Age 데이터 처리

def category_age(age):
    return age//10
    
train_df['cate_age'] = train_df.Age.apply(category_age)

test_df['cate_age'] = test_df.Age.apply(category_age)

# Sir 데이터 처리

def unique_sir(df):
    for data in enumerate(df.Sir.unique()):
        df.Sir.replace(data[1], data[0], inplace=True)
  
unique_sir(train_df)
unique_sir(test_df)

# Embarked 데이터 처리

def unique_emb(df):
    for data in enumerate(df.Embarked.unique()):
        df.Embarked.replace(data[1], data[0], inplace=True)
        
unique_emb(train_df)
unique_emb(test_df)

# Sex 데이터 처리

def unique_sex(df):
    for data in enumerate(df.Sex.unique()):
        df.Sex.replace(data[1], data[0], inplace=True)
        
unique_sex(train_df)
unique_sex(test_df)
```

---

6. **컬럼 삭제**

```python
# Survived와 관련 없는 컬럼
# 결측치가 대부분인 컬럼
# 한가지 값이 대부분인 컬럼

train_df.drop(['Name', 'Age', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['Name', 'Age', 'Ticket', 'Cabin'], axis=1, inplace=True)
```

---

7. **분류 - Decision Tree**

```python
y_label = train_df.Survived
train_df.drop('Survived', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df, y_label, train_size=0.99, test_size=0.01, random_state=4564561)

from sklearn.tree import DecisionTreeClassifier

dtclf = DecisionTreeClassifier()

dtclf = dtclf.fit(X_train, y_train)

dt_pred = dtclf.predict(X_val)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_val, dt_pred, labels=[1,0])

from sklearn.metrics import accuracy_score
val_accuracy = accuracy_score(y_val, dt_pred)

from sklearn.metrics import accuracy_score
val_accuracy = accuracy_score(y_val, dt_pred)

result = dtclf.predict(test_df)

test_dt = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':result})

test_dt.to_csv('test_dt.csv')
```

---

8. **분류 - RandomForest**

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10000)

rfc.fit(X_train, y_train)
pred = rfc.predict(X_val)

rfc.score(X_val, y_val)

pred = rfc.predict(test_df)

test_rf = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':pred})

test_rf.to_csv('test_rf.csv')
```

---

9. **분류 - DeepLearning**

```python
X = train_df
y = y_label

import keras
import os
from keras.models import Sequential
from keras.layers import Dense

os.environ["cuda_visible_devices"]= '0'

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=10))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
model.summary()

model.fit(X, y, epochs=6000)

pred = model.predict(test_df)
final = (pred > .5).astype(int).reshape(test_df.shape[0])
final = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':final})
final.to_csv('test_dl.csv', index=False)
```



