+++
title = "Art of categorical encoding"
description = "Talk:Art of categorical encoding"
date = 2018-06-24T02:13:50Z
author = "Shubh Chatterjee"
+++


## The below tutorial is also available as a talk that I gave at PyconAU 2020(If you prefer video to text mode.Below is the talk)

[![PyconAU-2020 Talk](http://img.youtube.com/vi/C3u4ip97wJo/0.jpg)](http://www.youtube.com/watch?v=C3u4ip97wJo "Categorical encoding for tabular data")


# Introduction

![](/images/pyconau2020/img/Introduction.jpg)

## Agenda of this talk

**Below is a typical simplified Machine learning model development pipeline for tabular data**

![](/images/pyconau2020/img/fig1.png)

In a typical model development pipeline there is raw data that exists (across servers/schemas etc) which is aggregated to get the exhaustive model development data or data which might be useful to solve the problem at hand .Post this the model development data is used to develop an outcome or the target variable(example:Sales,default,fraud etc) and independent variables which might be useful in predicting the target .The supervised machine learning algorithm uses the independent predictors and the target to develop a predictive entity which helps in getting an estimation for the predictive problem.

**Today's talk is based on how the raw variables(specifically categorical variables) should be transformed for usage into model development for better predictive accuracy and long term maintainance** 

https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-numerical-variables/

## A Quick segway into model development data types
<a id=data_types></a>

**Model development data** : Data captured for most of the problem statments that you might be trying to solve should fall in one of the below buckets:

- **Categorical variables:**A variable which does not represent a numeric entity or an entity that cannot be represented on a coordinate scale.They need to be transformed into a numeric format for usage in mathematical algorithms
 - **Ordinal variables:**variable  with inherent ranking/ordering
   - Examples:Academic grades(A++,A,A-,..),Age Bracket(New born,Baby,Toddler..)
 - **High cardinality:**variable with unique values which are greater than 15(**My own thumb rule**)
   - Examples: zipcodes,product IDs,Operating system version numbers,Email_domain_address
 - **Low cardinality:**variable with unique values which are less than 15(**My own thumb rule**)
   - Examples: credit_default_status(YES/NO),customer_status(Active/inactive/attrited)
 - **Variables that you might mistake to be numeric variables:**A variable whose values are numbers but does not have an inherent ordering to them
   - Examples : zipcodes,House-numbers,OS version numbers
- **Numeric variables:** A variable which can be represented as a numeric entity or on a coordinate scale.
 - The values that a numeric variable might take might vary depending upon the variable type and can be contiguous,integers,binary.They can be used directly as predictors in mathemarical algorithms
    - Examples :Distance,speed,Income,credit score,Indicator_for_having_a_pet(1/0)
- **Alternate data types**
 - **Text**
 - **Images**
 - **Videos**
 - **Audio**
 - **Every other damn thing under the blue sky** 🙄

Lets pick up an extremely popular dataset from kaggle to get a feel of the variable types we just encountered.


```python
# Titance dataset:Predict survival on the Titanic
#(An extremly popular and a kind of Hello world dataset within competitive predictive modelling landscape)

import pandas as pd
pd.options.mode.chained_assignment = None
df=pd.read_csv("data/train.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



In the dataset we have 12 variables, within which Survived(Who survives the Titanic) is the binary outcome to be predicted.Let's classify each of the other features into one of the above variable classification.


One of the quick ways to identify  a variables type other than business/domain knowledge is to check the data types of variable


```python
df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object



Its also worth checking the number of unique values for each variable


```python
df.shape
```




    (891, 12)




```python
df.nunique()
```




    PassengerId    891
    Survived         2
    Pclass           3
    Name           891
    Sex              2
    Age             88
    SibSp            7
    Parch            7
    Ticket         681
    Fare           248
    Cabin          147
    Embarked         3
    dtype: int64



We observe that there are four variables which are classified as object dtype  which is a hint that they are categorical variables(Name,Sex,Cabin,Embarked).
Further,we have two variables which have same number of unique values as number of passengers,indicating they are the primary keys.
Based on that we can classify the 12 variables as:

1.  **PassengerId :** The primary key passenger ID is a High cardinality categorical variable.Although the variable is numeric in form we are not classifying it as numeric as it cannot be used on numeric scale that is passenger ID 1 passenger ID 2 has no meaning.

2. **Survived :** The outcome variable as this is a classification problem is a binary numeric variable(I am classifying it as numeric as it in already encoded as 1/ 0 if it was survived/Not_survived it would have been a low cardinality categorical variable which we would have needed to transform into numeric binary form for development of a classification algorithm

3. **Pclass :** A Low cardinality categorical variable

4. **Name :** A High cardinality categorical variable

5. **Sex :** A Low cardinality categorical variable

6. **Age :** A Numeric variable

7. **SibSp :** # of siblings / spouses aboard the Titanic, A Numeric variable

8. **Parch:** # of parents / children aboard the Titanic,A Numeric variable

9. **Ticket:** Ticket number,A High cardinality categorical variable

10. **Fare:** Passenger fare,A Numeric Variable

11. **Cabin:** Cabin number,A High cardinality categorical variable

12. **Embarked:** Port of Embarkation,A Low cardinality categorical variable



# Transforming categorical variables

## Some guidelines around choosing a categorical variables tranformation methodology

As we mentioned when we were lookinng at typical data types we would face during a predictive development task that categorical variables in their raw form are not usable in a mathematical predictive algorithm and they need to be transformed into a numeric form.

There are various methodologies to conduct the above tranformation for the categorical variables but before we look at them lets define few guidelines around what our final product should be and how we might want to evaluate the results of transformation from categorical to numeric.Below are three major questions that we would ask to evaluate any categorical variable transformation methodology we might find.

- **How much incremental improvement we observe in models predictive strength compared to random prediction?**
- **Will the categorical variable transformation methodology be  supported by the technical infrastructure in place for inference of the model in production?**
- **How robust is the methodology against domain shift that we might observe in the data,which would eventually happen in this ever fluctuating world?**

## Methodologies for categorical variable transformation

A highly researched area within the Data science and machine learning community there are multiple ways to encode a categorical method.The best practices varies based upon data type,domain and compute power at disposal but  Below is a list of major variable transformation methodologies used in the Industry:

- **One Hot encoding**
- **Vanilla Count encoding**
- **Vanilla target encoding**
- **Vanilla Weight of evidence**


Let's delve into each of them using the Titanic dataset that we encountered in [Section-2](#data_types)


## One Hot encoding

In one hot encoding we transform the categories within the variable into their own individual binary representation.Below example will make it clear.

<u>Below is the Titanic dataset</u>


```python
print("The size of the dataset is {} with {} columns".format(df.shape[0],df.shape[1]))
```

    The size of the dataset is 891 with 12 columns



```python
df['Survived'].value_counts(normalize=True)
```




    0    0.616162
    1    0.383838
    Name: Survived, dtype: float64



<b> The Survival rate is 38% as per the training data<b>


```python

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



<b>Before we proceed we will do some Hygiene transformaton on the data like removing redundant variables,imputing missing values and some other sanity changes

<b> First lets split the data intp train and test 


```python
from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.2,random_state=2)
```

<b> Quick Basic Missing value imputation for few variables


```python
df_train['Age'].fillna(df_train['Age'].median(), inplace = True)
df_test['Age'].fillna(df_train['Age'].median(), inplace = True)

df_train['Embarked'].fillna(df_train['Embarked'].mode().iloc[0], inplace = True)
df_test['Embarked'].fillna(df_train['Embarked'].mode().iloc[0], inplace = True)

df_train['Cabin'].fillna(df_train['Cabin'].mode().iloc[0], inplace = True)
df_test['Cabin'].fillna(df_train['Cabin'].mode().iloc[0], inplace = True)

df_train['Pclass']=df_train['Pclass'].astype('object')
df_test['Pclass']=df_test['Pclass'].astype('object')
```

<b>Let's transform the port of embarkment of the passengers using one Hot encoding,where we will have seperate binary representation for each embarkment type


```python
df_train['Embarked'].unique() # C = Cherbourg, Q = Queenstown, S = Southampton
```




    array(['C', 'S', 'Q'], dtype=object)




```python
df_train.join(pd.get_dummies(df_train['Embarked'],prefix='Embarked'))[['Embarked','Embarked_C','Embarked_Q','Embarked_S']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Q</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We actually need only n-1 categories to be binarized that is Embarked_S,Embarked_Q in itself capture if Embarked_C exists or not.Hence:


```python
df_train.join(pd.get_dummies(df_train['Embarked'],prefix='Embarked',drop_first=True))[['Embarked','Embarked_Q','Embarked_S']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>C</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Q</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We would now convert all predictive categorical variables in the dataset into One-hot encoded form and would attempt to develop a quick ML algorithm to predict the survival

<b>First a List of all Categorical variables in the Titanice dataset which heuristically could be predictors of survival of a passenger


```python
Cat_predictors=list(df.drop(['PassengerId','Survived','Ticket',"Name","Age","Fare","SibSp","Parch"],axis=1).columns)

```


```python
Cat_predictors
```




    ['Pclass', 'Sex', 'Cabin', 'Embarked']



<b>The Below one liner in pandas will one hot encode all variables in Cat_predictors<b>


```python
import category_encoders as ce

OHE=ce.OneHotEncoder(df_train[Cat_predictors],use_cat_names=True)

OHE.fit(df_train[Cat_predictors])

df_train_OHE=df_train.join(OHE.transform(df_train[Cat_predictors]))
df_test_OHE=df_test.join(OHE.transform(df_test[Cat_predictors]))

```

Time to develop a quick model and check the predictive quality .For Absolute simplicity for this classification problem we will use a Logistic regression model .There would be probably lots of sighs and roll of eyes 🙄 but come on folks this is a toy problem,we are not trying to beat SOTA 😉

<b> <font color='red'> Note:There is lots of hand waving in the model development steps ignoring steps like correlations,robust missing value imputation,hyperparameter tuning and many other fine factors which might influence the scientific quality of a predictive model.We are doing that to be able to capture the flavor of categorical encoding within the stipulated time period.A model development process is a very nuanced process a combination of art and science.</font>
    


```python
%load_ext autoreload
%autoreload 2
from utils import *
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


<b> For simplicity all the intermediate steps of model training have been pushed into a utils function,which we would be using heavily in the document


```python
model_and_predict(df_train_OHE,df_test_OHE,Cat_predictors)
```

    Training AUC is 0.8836239382827915,
    Test AUC is 0.8316455696202532
    Incremental improvement over random is 0.5721464465183058


- **How much incremental improvement we observe in models predictive strength?**
  - ~57% above random prediction
- **Will the categorical variable transformation methodology be  supported by the technical infrastructure in place for inference of the model in production?**
   - The transformation process is quite simple computationally but the write cost of categorical to binary can be quite high in case of high cardanilty variables.For example:In the Titanic dataset we have 147 cabin numbers ,Hence one variable got transformed into 146 different binary variable
- **How robust is the methodology against domain shift that we might observe in the data,which would eventually happen in this ever fluctuating world?**
   - In case there is a domain shift in a variable,we might loose information capture.Example: If we are applying the Titanic classification model to lets say another ship disaster ,lets call it Thanos and in case in Thanos the cabin numbers are similar but there are 150 cabin numbers instead of 147 the one hot encoded variables will not be able to capture the details about the other three.
   <b> A solution to make the encoding stable with respect to a domain shift is including a catch-all binary variable which captured every other category that might pop-up in the future inference data<b>



## Vanilla Count encoding

In count encoding every category within the variable is replaced with its corresponding count.The counts for each categiry is stored and used to encode variables during inference or in production

<b> We will set up the count encoders and then fit it on training data


```python
count_enc=ce.CountEncoder(cols=Cat_predictors)

count_enc.fit(df_train[Cat_predictors])
```




    CountEncoder(cols=['Pclass', 'Sex', 'Cabin', 'Embarked'],
                 combine_min_nan_groups=True, drop_invariant=False,
                 handle_missing='count', handle_unknown=None, min_group_name=None,
                 min_group_size=None, normalize=False, return_df=True, verbose=0)



<b>Let's have a look at the transformation of the Variables


```python
df_train.join(count_enc.transform(df_train[Cat_predictors]).add_suffix('_count'))[['Pclass','Pclass_count']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Pclass_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>175</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>389</td>
    </tr>
    <tr>
      <th>873</th>
      <td>3</td>
      <td>389</td>
    </tr>
    <tr>
      <th>182</th>
      <td>3</td>
      <td>389</td>
    </tr>
    <tr>
      <th>876</th>
      <td>3</td>
      <td>389</td>
    </tr>
    <tr>
      <th>213</th>
      <td>2</td>
      <td>148</td>
    </tr>
    <tr>
      <th>157</th>
      <td>3</td>
      <td>389</td>
    </tr>
    <tr>
      <th>780</th>
      <td>3</td>
      <td>389</td>
    </tr>
    <tr>
      <th>572</th>
      <td>1</td>
      <td>175</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3</td>
      <td>389</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train['Pclass'].value_counts()
```




    3    389
    1    175
    2    148
    Name: Pclass, dtype: int64



<b>As the results are what we want we will transform the train and test data for usage in model development


```python
df_train_vce=df_train.join(count_enc.transform(df_train[Cat_predictors]).add_suffix('_count'))
df_test_vce=df_test.join(count_enc.transform(df_test[Cat_predictors]).add_suffix('_count'))

```

<b> This looks farely simple but there are some finer points which would come to haunt you during Inference time or during time you are productionalizing the model

<b> What happens if there is a domain shift,that is what if it is applied to a ship which might have cabin numbers that differ?Luckily we have the example simulated here in our test set.There are cabin numbers in test set which are not there in train set


```python
list(set(df_test['Cabin'].unique())-set(df_train['Cabin'].unique()))
```




    ['A32',
     'D9',
     'B42',
     'D7',
     'D20',
     'C148',
     'A16',
     'B73',
     'E36',
     'A23',
     'A31',
     'D45',
     'E17',
     'C30',
     'E50',
     'F4',
     'E63',
     'F E69']



<b> How are these Cabin numbers encoded in the test set?


```python
df_test_vce[['Cabin','Cabin_count']].loc[df_test_vce['Cabin_count'].isna()]
```


```python
df_test_vce[['Cabin','Cabin_count']].loc[df_test_vce['Cabin'].isin(list(set(df_test['Cabin'].unique())-set(df_train['Cabin'].unique())))].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cabin</th>
      <th>Cabin_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>A23</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>128</th>
      <td>F E69</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>185</th>
      <td>A32</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>209</th>
      <td>A31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>520</th>
      <td>B73</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



<b> Yes,so Nan as the count encoder has not seen these values in the encoding data within training dataset.Now,what do we do?
- The encoding code should have an additional handler for this as we will always observe these as the domain shifts and we move deeper into inference time period
- There are multiple ways we can handle it,for simplicity we will impute the unknowns by -9999


```python
df_test_vce.fillna(-9999,inplace=True)

df_test_vce[['Cabin','Cabin_count']].loc[df_test_vce['Cabin'].isin(list(set(df_test['Cabin'].unique())-set(df_train['Cabin'].unique())))].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cabin</th>
      <th>Cabin_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>A23</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>128</th>
      <td>F E69</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>185</th>
      <td>A32</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>209</th>
      <td>A31</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>520</th>
      <td>B73</td>
      <td>-9999.0</td>
    </tr>
  </tbody>
</table>
</div>



<b> So,we have transformed the train and test into the format we wanted  using count encoding,Lets quickly develop  model so that we can look at the results


```python
model_and_predict(df_train_vce,df_test_vce,Cat_predictors)
```

    Training AUC is 0.8657218830184525,
    Test AUC is 0.8384810126582278
    Incremental improvement over random is 0.5850681981335247


- **How much incremental improvement we observe in models predictive strength?**
  - ~58% above random prediction and better than one hot encoding.In practice in most cases you will find count encoding has bettter prediction numbers than a simple one hot encoding but there is also a higher possibility of overfitting(There are ways around it we will discuss it soon)
- **Will the categorical variable transformation methodology be  supported by the technical infrastructure in place for inference of the model in production?**
  - A simple process of transformation and much lower write cost and variable maintainance cost compared to one hot encoding as each variable is represented by one variables post transformation compared to OHE where each variable is replaced with close to as many categories as in the variable
- **How robust is the methodology against domain shift that we might observe in the data,which would eventually happen in this ever fluctuating world?**
   - We have discusssed how to handle domain shift while doing count encoding
 

## Vanilla target encoding

In Target encoding each category within a variable is represented by the summary of target/outcome that it captures.

<b> Lets set up a target encoder and check how the results look like


```python
Target_enc=ce.TargetEncoder(cols=Cat_predictors)

Target_enc.fit(df_train[Cat_predictors],df_train['Survived'])
```




    TargetEncoder(cols=['Pclass', 'Sex', 'Cabin', 'Embarked'], drop_invariant=False,
                  handle_missing='value', handle_unknown='value',
                  min_samples_leaf=1, return_df=True, smoothing=1.0, verbose=0)




```python
df_train_vte=df_train.join(Target_enc.transform(df_train[Cat_predictors]).add_suffix('_target'))
df_test_vte=df_test.join(Target_enc.transform(df_test[Cat_predictors]).add_suffix('_target'))

```


```python
df_train_vte.groupby('Embarked')['Survived'].agg('mean')
```




    Embarked
    C    0.513889
    Q    0.367647
    S    0.328000
    Name: Survived, dtype: float64




```python
df_train_vte[['Embarked','Embarked_target']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Embarked_target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>C</td>
      <td>0.513889</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S</td>
      <td>0.328000</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Q</td>
      <td>0.367647</td>
    </tr>
  </tbody>
</table>
</div>



<b> Here,we have replace variable sex by the mean captured by rolled up categories within the variable

<b> Target encoding within category encoders by default replaces the unknowns in the test by the base rate of outcome.Which again is a very rudimentray way of handling this.It can be handled in multiple different ways based on domain knowledge and EDA


```python
df_test_vte[['Cabin','Cabin_target']].loc[df_test_vte['Cabin'].isin(list(set(df_test['Cabin'].unique())-set(df_train['Cabin'].unique())))].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cabin</th>
      <th>Cabin_target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>A23</td>
      <td>0.369382</td>
    </tr>
    <tr>
      <th>128</th>
      <td>F E69</td>
      <td>0.369382</td>
    </tr>
    <tr>
      <th>185</th>
      <td>A32</td>
      <td>0.369382</td>
    </tr>
    <tr>
      <th>209</th>
      <td>A31</td>
      <td>0.369382</td>
    </tr>
    <tr>
      <th>520</th>
      <td>B73</td>
      <td>0.369382</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_and_predict(df_train_vte,df_test_vte,Cat_predictors)
```

    Training AUC is 0.8711331475945701,
    Test AUC is 0.810379746835443
    Incremental improvement over random is 0.5319454414931801


- **How much incremental improvement we observe in models predictive strength?**
  - We observe a high overfitting when we use Target encoding.Hence,a drop in AUC compared to previous methods but it can be handled well as we will observe soon
- **Will the categorical variable transformation methodology be  supported by the technical infrastructure in place for inference of the model in production?**
  - A simple process of transformation and much lower write cost and variable maintainance cost compared to one hot encoding as each variable is represented by one variables post transformation compared to OHE where each variable is replaced with close to as many categories as in the variable
- **How robust is the methodology against domain shift that we might observe in the data,which would eventually happen in this ever fluctuating world?**
   - It has to be figured out how you want to handle the categories introduced due to domain shift,here we have handled using base rate of outcome in the training data
 

## Vanilla Weight of evidence

The weight of evidence(WOE) can be defined as log to the ratio of percentage of events to percentage of non events  within the category being encoded for a variable.It will become easier to understand as we look at an example.We will pick Sex as a variable to learn more about weight of evidence


```python
WOE_enc=ce.WOEEncoder(cols='Embarked',regularization=0) #regularization is zero to replicate the WOE manually
WOE_enc.fit(df_train['Embarked'],df_train['Survived'])
```




    WOEEncoder(cols=['Embarked'], drop_invariant=False, handle_missing='value',
               handle_unknown='value', random_state=None, randomized=False,
               regularization=0, return_df=True, sigma=0.05, verbose=0)




```python
df_train.join(WOE_enc.transform(df_train['Embarked']).add_suffix('_WOE'))[['Embarked','Embarked_WOE']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Embarked_WOE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>C</td>
      <td>0.590439</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S</td>
      <td>-0.182376</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Q</td>
      <td>-0.007455</td>
    </tr>
  </tbody>
</table>
</div>



<b> To understand the calculations underneath let's calculate the WOE overself


```python
def calc_WOE(df,var,outcome):
    overall_number_of_ones = df[outcome].sum()
    overall_number_of_zeroes=df.shape[0]-overall_number_of_ones
    grouped = pd.DataFrame()
    grouped['Total'] = df.groupby(var)[outcome].agg('count')
    grouped['number of ones'] = df.groupby(var)[outcome].agg('sum')
    grouped['number of zeroes'] = grouped['Total'] - grouped['number of ones']

    grouped['percentage of ones'] = grouped['number of ones'] / overall_number_of_ones
    grouped['percentage of zeroes'] = grouped['number of zeroes'] / overall_number_of_zeroes
    grouped['(% ones) > (% zeroes)'] = grouped['percentage of ones'] > grouped['percentage of zeroes']
    grouped['WOE']=np.log(grouped['percentage of ones']/grouped['percentage of zeroes'])
    
    return grouped

```


```python
calc_WOE(df_train,'Embarked','Survived')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>number of ones</th>
      <th>number of zeroes</th>
      <th>percentage of ones</th>
      <th>percentage of zeroes</th>
      <th>(% ones) &gt; (% zeroes)</th>
      <th>WOE</th>
    </tr>
    <tr>
      <th>Embarked</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>144</td>
      <td>74</td>
      <td>70</td>
      <td>0.281369</td>
      <td>0.155902</td>
      <td>True</td>
      <td>0.590439</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>68</td>
      <td>25</td>
      <td>43</td>
      <td>0.095057</td>
      <td>0.095768</td>
      <td>False</td>
      <td>-0.007455</td>
    </tr>
    <tr>
      <th>S</th>
      <td>500</td>
      <td>164</td>
      <td>336</td>
      <td>0.623574</td>
      <td>0.748330</td>
      <td>False</td>
      <td>-0.182376</td>
    </tr>
  </tbody>
</table>
</div>



<b> As we observe the WOE is a log of ratio of percentage of ones to percentage of zeros found under each variable level (female/male) in this case

<b> Lets now calcualte the WOE for the entire data


```python
WOE_enc=ce.WOEEncoder(cols=Cat_predictors) 
WOE_enc.fit(df_train[Cat_predictors],df_train['Survived'])
```




    WOEEncoder(cols=['Pclass', 'Sex', 'Cabin', 'Embarked'], drop_invariant=False,
               handle_missing='value', handle_unknown='value', random_state=None,
               randomized=False, regularization=1.0, return_df=True, sigma=0.05,
               verbose=0)




```python
df_train_WOE=df_train.join(WOE_enc.transform(df_train[Cat_predictors]).add_suffix('_WOE'))
df_test_WOE=df_test.join(WOE_enc.transform(df_test[Cat_predictors]).add_suffix('_WOE'))

```

<b> What are we doing about new categories being introduced in the test data?


```python
df_test_WOE[['Cabin','Cabin_WOE']].loc[df_test_WOE['Cabin'].isin(list(set(df_test['Cabin'].unique())-set(df_train['Cabin'].unique())))].head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cabin</th>
      <th>Cabin_WOE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>A23</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>128</th>
      <td>F E69</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>185</th>
      <td>A32</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>209</th>
      <td>A31</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>520</th>
      <td>B73</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We are replacing the new categories by a weight of evidence of zero,which is fair as the training data does not have any information about them


```python
model_and_predict(df_train_WOE,df_test_WOE,Cat_predictors)
```

    - Training AUC is 0.8748930873000413,
    - Test AUC is 0.820886075949367
    - Incremental improvement over random is 0.5518066523091647


- **How much incremental improvement we observe in models predictive strength?**
  - As this method also use the target for transformation of categorical variables similar to Target encoding we observe a high overfitting (Big fall in discrimination performance when we move from train to test)
- **Will the categorical variable transformation methodology be  supported by the technical infrastructure in place for inference of the model in production?**
  - A simple process of transformation and much lower write cost and variable maintainance cost compared to one hot encoding as each variable is represented by one variables post transformation compared to OHE where each variable is replaced with close to as many categories as in the variable
- **How robust is the methodology against domain shift that we might observe in the data,which would eventually happen in this ever fluctuating world?**
   - We assigned a predictive WOE of zero that is neither it positively impacts survival nor negatively as per training data.Again,as per the domain knowledge this can be replaced by something else
 

## Some other useful methods

<b>There are some other methods but the usage of them depends on the domain and type of variable you might be working on.If you reckon it fits your use case they are worth trying out to attemot the predictive efficiency of the  task

- Ordinal encoding 
 - <b>When the variable is ordinal or has a hierarchy to it the levels within the variable can be represented by 
   integers(Difference between each integer can be chosen as per the heuristic difference in levels).You might also benefit with ordinal encoding instead of one hot encoding in some non-hierarchial instances (if the hierarchy might represent something that model might pick up)
- BaseN encoding
 - <b> Encode different level of categories into their BaseN representation,for example if N=1 it will be one hot encoded,if N=2 it will be binary encoded(Similar to OHE but we have a level 00)
- Contrast coding
 - <b> Useful when you want to encode a variable with respect to the difference to the previous level.Useful mainly for ordinal varaibles,in practice if applied well can improve the predictive strength of a model
- Jame-stein encoding
 - <b> A variance of Target encoding,while the target encoding replaces each variable level with its Target mean.This method further normalizes it using the  base rate of the population.There are weights of category mean and population mean which can be used to fine tune the encoding.Helps in reducing overfitting
- Entity Embedding:
    <b> A very useful and often overlooked method,where you would encode each of the category within a variable into the Euclidean space by using Neural nets (Highly recommend this fantastic paper                                  [Entity Embeddings of Categorical Variables](https://arxiv.org/pdf/1604.06737.pdf) where the authors used Entity embeddings to reach third position in a kaggle competition)




# Pragmatic workarounds for problems in the wild

![](/images/pyconau2020/img/reality.gif)

<b> Although we have some good methods to try out to encode the categorical variables.As it is always across everything that we might want to do or learn,Excellence is in detail and the detail always boils down to how the data is in reality

In practice there are various situations that you might encounter,few of which we have already observed at a very small scale:

## Cross validating the category encoding when the method overfits


Most of the supervised encoding techniques(Target encoding,weight of evidence,Jame-stein encoding) will overfit the training data as the strength of encoding is highly dependent on how good a representation is the training data is to the in the wild population.For example:If in the training data while encoding gender we observe that males have lower survival rates than femals but the observations is opposite on an inference dataset.The encoding and simultanously the model will be negatively impacted.A way to resolve it is <b>cross validating the encoding. 

<b> The encoding values would be calculated from the corresponding cross validation folds, Hence,there will be diversification of values for each category leading to less chances of overfitting


```python
from sklearn.model_selection import KFold, StratifiedKFold
def Target_encoder_kfold(train_data,test_data,Cat_vars,outcome,n_folds=4):
    """
    Target encode all the cat/object vars in the data using 
    
    Attributes
    ----------
    
    train_data:Pandad Dataframe,The data which is to be used the develop the model
    test_data: Pandas Dataframe,The test data for model evaluation
    outcome: Str,The string variable name which represents the outcome
    n_folds:Number of folds for cross validation,default is 4
    
    Returns
    --------
    train_kfold_cp=The training data where each cat/obj var has been replaced by taget encode variable,suffix _target
    test_kfold_cp=The test data where each cat/obj var has been replaced by taget encode variable,suffix _target
    
    
    
    """
    
    train_kfold_cp=train_data.copy()
    test_kfold_cp=test_data.copy()
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=1001)
    for cols in Cat_vars :
        train_kfold_cp[cols]=train_kfold_cp[cols].astype('str')
        test_kfold_cp[cols]=test_kfold_cp[cols].astype('str')
        #print(cols)
        train_enc = np.zeros((train_data.shape[0]))
        for train_index,test_index in folds.split(train_kfold_cp):        

            #print(f"train : {train_R.iloc[train_index].shape} & test :{train_R.iloc[test_index].shape}")

            df_mean=train_kfold_cp.iloc[train_index].groupby(by=cols,as_index=False)[outcome].mean()

            df_mean.columns=['ID',cols+'_target']

            train_enc[test_index]=pd.merge(train_kfold_cp.iloc[test_index],df_mean,how='left',left_on=cols,right_on='ID')[cols+'_target'].to_numpy()


            #target_enc = ce.TargetEncoder(cols=cols,handle_missing='return_nan',min_samples_leaf=0,smoothing=0)
            #target_enc.fit(train_R.iloc[train_index][cols], train_R.iloc[train_index]['target'])
            #print(train_R.iloc[test_index][cols].head())
            #print(cols)
            #train_enc[test_index]=target_enc.transform(train_R.iloc[test_index][cols]).to_numpy()
        train_kfold_cp[cols+'_target']=train_enc
        df_mean=train_kfold_cp.groupby(by=cols,as_index=False)[outcome].mean()
        df_mean.columns=['ID',cols+'_target']
        test_kfold_cp=pd.merge(test_kfold_cp,df_mean,how='left',left_on=cols,right_on='ID').drop('ID',axis=1)
        #del train_kfold_cp[cols]
        #del test_kfold_cp[cols]

        print(f"Done with {cols}")
    return train_kfold_cp,test_kfold_cp 
    


```


```python
df_train_kfold,df_test_kfold=Target_encoder_kfold(df_train,df_test,Cat_predictors,'Survived',n_folds=4)
```

    Done with Pclass
    Done with Sex
    Done with Cabin
    Done with Embarked


<b> Lets have a quick look at how encoding has been done


```python
df_train_kfold[['Embarked','Embarked_target']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Embarked_target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>C</td>
      <td>0.473214</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S</td>
      <td>0.324607</td>
    </tr>
    <tr>
      <th>873</th>
      <td>S</td>
      <td>0.327177</td>
    </tr>
    <tr>
      <th>876</th>
      <td>S</td>
      <td>0.335150</td>
    </tr>
    <tr>
      <th>213</th>
      <td>S</td>
      <td>0.325269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>0.537736</td>
    </tr>
    <tr>
      <th>484</th>
      <td>C</td>
      <td>0.514851</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Q</td>
      <td>0.346939</td>
    </tr>
    <tr>
      <th>330</th>
      <td>Q</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>111</th>
      <td>C</td>
      <td>0.530973</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Q</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>612</th>
      <td>Q</td>
      <td>0.387755</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train_kfold.groupby('Embarked')['Embarked_target'].nunique()
```




    Embarked
    C    4
    Q    4
    S    4
    Name: Embarked_target, dtype: int64



<b> As we observe,each of the categories seems to have multiple encoded values,that is there are 4 encoding(from each of the folds ) for each category

<u><i> That is how the training data is encoded,a quick look at the test data or the inference data


```python
df_test_kfold[['Embarked','Embarked_target']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Embarked_target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S</td>
      <td>0.328000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>C</td>
      <td>0.513889</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Q</td>
      <td>0.367647</td>
    </tr>
  </tbody>
</table>
</div>



<b> The test data is encoded using a vanilla target encoding.It solved two purpose:
- Reduces overfitting in the wild as we have variance on trainig and test data during model development
- A simpler inference workflow

<b> Handling few edge cases:
- For all those categories within a variable where we have only 1 observation we make it NA or Nan because we cannot estimate the target propensity of those using just 1 observation.This is done in both train and test set
- This helps to reduce overfitting
    
<u> The missing values can be handled using advanced Missing value imputation techniques so that we have better estimate for their propensity.For simplicity we will right now replace them by -9999


```python
df_train_kfold.loc[df_train_kfold['Cabin_target'].isna()].groupby('Cabin')['Cabin'].nunique()
```




    Cabin
    A10      1
    A14      1
    A19      1
    A20      1
    A24      1
            ..
    E77      1
    F G63    1
    F2       1
    F38      1
    T        1
    Name: Cabin, Length: 108, dtype: int64




```python
df_train_kfold.fillna(-9999, inplace = True)
df_test_kfold.fillna(-9999, inplace = True)
```


```python
model_and_predict(df_train_kfold,df_test_kfold,Cat_predictors)
```

    Training AUC is 0.8656541363570927,
    Test AUC is 0.8156962025316457
    Incremental improvement over random is 0.5419956927494618



```python
(0.8156962025316457-0.810379746835443)/0.810379746835443
```




    0.006560449859419076



<b> Even for this basic example we can observe if we compare the K-fold Target encoding with Vanilla encoding there has been a reduction in overfitting and hence an impovement in test AUC from 0.810379746835443 to 0.8156962025316457.The K-fold CV methodology explained above should be used to enhance any Category encoding method(Count encoding,Weight of evidence etc).In almost all cases you will always observe a reduction in overfitting 

## Some final thumb rules/practioners tips

- <b>Never pre-optimize<b> :The first target should always be getting a baseline model to which any enhancment should be compared.Always start with something simple and build on top of it
- <b> There is no perfect encoding technique<b> : Different use cases might lead to different techniques proving better results.It highly depends on the domain and data


<b><font color='blue'> That's it! Thanks for your time.It was a pleasure.I can be found floating around the world wide web on below two platforms.
   
- Twitter-[@shub777](https://twitter.com/shub777)
- Linkedin-[Shubrashankh Chatterjee](https://www.linkedin.com/in/shubrashankh-chatterjee/)
- Github-https://github.com/shubchat
    
    
If you want to play around with the notebook in the talk you will find it at:
    
-https://github.com/shubchat/pyconau_2020_cat_encode
    

![](/images/pyconau2020/img/Thanks.gif)
