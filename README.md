# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs

Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
1. Importing the libraries

2. Importing the dataset

3. Taking care of missing data

4. Encoding categorical data

5. Normalizing the data

6. Splitting the data into test and train


## PROGRAM:
```python
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
df.head()
le=LabelEncoder()
df["CustomerId"]=le.fit_transform(df["CustomerId"])
df["Surname"]=le.fit_transform(df["Surname"])
df["CreditScore"]=le.fit_transform(df["CreditScore"])
df["Geography"]=le.fit_transform(df["Geography"])
df["Gender"]=le.fit_transform(df["Gender"])
df["Balance"]=le.fit_transform(df["Balance"])
df["EstimatedSalary"]=le.fit_transform(df["EstimatedSalary"])
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
print(df.isnull().sum())
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)
df.duplicated()
print(df['Exited'].describe())
scaler= MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
x_train,x_test,y_train,x_test=train_test_split(X,Y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
### Printing first five rows of the dataset:

![head](https://user-images.githubusercontent.com/93427086/229786624-434d3b27-76bb-470a-829d-e9233afcd3c7.png)

### Separating x and y values: 

![separate](https://user-images.githubusercontent.com/93427086/229786818-490df5c1-4c21-4dc7-9510-a314c5f128a6.png)

### Checking NULL value for the dataset:

![null](https://user-images.githubusercontent.com/93427086/229787000-0d104156-1183-479f-b365-889b3fbaa912.png)

### Column y and its description:

![coly](https://user-images.githubusercontent.com/93427086/229787322-1ed2793b-8dfe-41ac-8222-c0fef593af34.png)

### Applying data preprocessing technique and printing the dataset

![dataprepro](https://user-images.githubusercontent.com/93427086/229787545-36dc7404-35c4-41c4-819f-7871aa29e115.png)

### Training Set

![train](https://user-images.githubusercontent.com/93427086/229787685-0dd4e6eb-14a3-4969-a3fd-ad2458f6839d.png)

### Testing Set and its length

![test](https://user-images.githubusercontent.com/93427086/229787948-ed15f323-3dbe-40f3-ab46-f2ab6a2065fd.png)


## RESULT
Hence the data preprocessing is done using the above code and data has been splitted into trainning and testing data for getting a better model.
