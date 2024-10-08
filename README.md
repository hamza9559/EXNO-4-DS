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
df= pd.read_csv('/content/bmi.csv')
df
```
![image](https://github.com/user-attachments/assets/8ba117dc-08ca-47a0-9478-d8ea598d45b8)


```
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
df[['Height','Weight']]= sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/42437bcd-70cd-46a6-a3bf-3b1558ae5083)

```
from sklearn.preprocessing import MinMaxScaler
mm= MinMaxScaler()
df[['Height','Weight']]= mm.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/b5b05440-557c-46d7-a973-b448e5c0d02b)

```
from sklearn.preprocessing import Normalizer
nm= Normalizer()
df[['Height','Weight']]= nm.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/866e8183-6d36-40f0-99f7-911a8a505caf)

```
from sklearn.preprocessing import MaxAbsScaler
mas= MaxAbsScaler()
df[['Height','Weight']]= mas.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/c462204e-5962-4d60-a208-729e4364e434)

```
from sklearn.preprocessing import RobustScaler
rs= RobustScaler()
df[['Height','Weight']]= rs.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/f49fcf1f-97c1-4138-9edf-ed159e630394)

# FEATURE SELECTION PROCESS

```
import pandas as pd
import  numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
df= pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
df
```

![image](https://github.com/user-attachments/assets/062af306-aa7f-43be-8b28-6e7559b47d8b)

```
df.isnull().sum()
```

![image](https://github.com/user-attachments/assets/dbde7cd7-05c7-4770-a020-c17fcf7965cf)

```
missing = df[df.isnull().any(axis=1)]
missing
```

![image](https://github.com/user-attachments/assets/d551d657-c9dc-469a-9985-d060edf70bc1)

```
df2= df.dropna(axis=0)
df2
```

![image](https://github.com/user-attachments/assets/6e4f7a11-c51d-4cbd-a362-79d353604b34)

```
sal= df['SalStat']
df2['SalStat']= df2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(df2['SalStat'])
```

![image](https://github.com/user-attachments/assets/2b1a73be-b5ee-4b21-8e28-88b7ffdffb6d)

```
sal2= df2['SalStat']
dfs= pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/1d8d4a1e-c158-432f-862c-20dd610e0e2b)

```
new_data= pd.get_dummies(df2,drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/58289ae6-3664-43f3-85ba-0156210ef9c1)

```
columns_list= list(new_data.columns)
print(columns_list)
```

![image](https://github.com/user-attachments/assets/67480def-b085-47ef-82a6-a0c4e648a134)


```
y= new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/2d971f93-32db-4fd2-a0a9-e941d5612546)

```
x= new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/dcbb910b-eb67-4822-b2e4-790a1e6719df)

```
train_x,test_x,train_y,test_y= train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier= KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/user-attachments/assets/5ec65caf-0b56-43bf-8865-9f3eb2f9d64d)

```
prediction= KNN_classifier.predict(test_x)
confusion_matrix(test_y,prediction)
```

![image](https://github.com/user-attachments/assets/dac6bfee-008e-4f93-a0cd-3c342183004f)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
df= sns.load_dataset('tips')
df.head()
```

![image](https://github.com/user-attachments/assets/f0296fd3-dfe6-4836-980f-909146bf2c04)

```
contingency_table= pd.crosstab(df['sex'],df['time'])  
print('contingency_table :-\n',contingency_table)
```

![image](https://github.com/user-attachments/assets/d9644ba0-12e5-488a-9838-4aed7ff1f800)

```
chi2, p,_,_= chi2_contingency(contingency_table)
print('chi-square statistic:',chi2)
print('p-value:',p)
```

![image](https://github.com/user-attachments/assets/8797b665-03f5-4457-a4ce-aba806d2a59b)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data= {
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target': [0,1,1,0,1]
}
df= pd.DataFrame(data)
X= df[['Feature1','Feature3']]
Y= df['Target']
selector= SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,Y)
selected_feature_indices= selector.get_support(indices=True)
selected_features= X.columns[selected_feature_indices]
print('Selected Features:\n',selected_features)
```

![image](https://github.com/user-attachments/assets/56f8d21c-71e4-49b3-b85c-1c76563a1fcd)


# RESULT:
       Hence,Feature Scaling and Feature Selection process has been performed on the given data set. 
