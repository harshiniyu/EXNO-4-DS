# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
  ```
import pandas as pd
from scipy import stats
import numpy as np
```
 ```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/3b2a5f35-1da0-4c9c-a519-5a09728d1191)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/089ea31c-080f-4274-9b11-de9500d3bd38)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/5df8a656-81ff-44ee-9040-59a7f253c71f)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/2492ee83-057f-4bbf-ac02-5b9958df8b83)
```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/user-attachments/assets/bfab4886-f911-4e97-9866-dd2d71d3345e)
```
sc=StandardScaler()
```
```
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/6ed65291-69a8-4e51-8eea-fb820e2ef222)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/9528eb14-f09e-4e64-9f4b-899a86bab32a)
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/21dd22b9-b2ab-4895-b5d2-9f27fbf2262d)
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![image](https://github.com/user-attachments/assets/5ee5d358-2b6e-4bd3-91dd-c19d6f843163)
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/44a038fd-3b53-4447-bdbd-2dd1378fe32e)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/819a85c8-a819-423f-930e-3ff12e9d7296)
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/67dcc769-70ae-46db-9fe9-9aa572b34572)
```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/3796abac-f0bf-49b7-a68c-d5d71377fc9e)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/5edcda58-a695-4567-8f98-a9c64c609b1d)

```
y_pred = rf.predict(X_test)
```
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```

![image](https://github.com/user-attachments/assets/4f51d468-49a3-4c6c-bc67-fdc4dc263760)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/617d1716-1b1c-404c-9436-83c880101508)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![Screenshot 2025-04-29 151250](https://github.com/user-attachments/assets/82a9d8ab-e8a7-4655-b357-b7986b0a9ee9)


```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![Screenshot 2025-04-29 152526](https://github.com/user-attachments/assets/7b0f2c5b-fa3f-48ee-b4da-c83de56774e1)


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![Screenshot 2025-04-29 152203](https://github.com/user-attachments/assets/2f20aded-e65f-4735-9b86-943da2606b3c)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/591dea92-782a-4281-8da0-b8963f998f3a)
```
!pip install skfeature-chappers
```

![image](https://github.com/user-attachments/assets/f435f3e7-d6fa-408d-b0f5-92c9a25bbef8)
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/bc4375c8-ce1a-4e7d-9ddb-93021d591df6)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

```

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")

categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')

```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/9492663f-8605-486c-9198-5cb0b1e78ac9)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

![image](https://github.com/user-attachments/assets/b46706b3-fd07-45a3-bb6a-ca1f2b386233)


# RESULT:
      
Thus the feature scaling selection was successfully executed
