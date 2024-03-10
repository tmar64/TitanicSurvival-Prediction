# Titanic Survival Prediction Web Application

Trained a RandomForestModel to predict if you would survive on the Titanic.
Takes age, gender, and # of siblings as input.

Kaggle Dataset: https://www.kaggle.com/c/titanic/data

I WILL UPDATE THIS PROJECT AS I WORK ON IT

###

## IMPORTS
```ruby
from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
```

###

## Loading the data, lets take a look
```ruby
file_path = "/Users/tanaymarathe/PycharmProjects/titanic_learning_model/train.csv"
df = pd.read_csv(file_path)
print(df.head())

   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S
```

Off the bat, we see irrelavent columns

### 

Lets visualize some of this data
