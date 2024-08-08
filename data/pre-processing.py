from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# read in the data file (csv)
def read_file():
    file_path = "/Users/tanaymarathe/Desktop/titanic/train.csv"
    df = pd.read_csv(file_path)
    return df


# find frequency of missing values (age column)
def missing_value(df):
    missing_age_count = df.isnull(df['Age']).sum()
    return missing_age_count


# fill missing data from 'age column' using 'forward fill method'
def fill_age(df):
    df['Age'] = df['Age'].ffill()
    return df


# drop columns that show little to no correlation
def drop_column(df):
    df.drop([], inplace=True, axis=1)
    return df


# brute force build population pyramid
def population_pyramid(df):

    # create df of m/f and age
    sex_age = df[['Sex','Age']]

    # creates df of m and age
    male_age = sex_age[sex_age.Sex == 'male']

    # sort df in ascending order, by age
    male_age = male_age.sort_values(by=['Age'], ascending=True)

    # create series of age and their frequency
    male_age_frequency = male_age['Age'].value_counts(sort=False)

    age_freq = 0
    for age, freq in male_age_frequency.items():
        if age <= 5:
            age_freq += freq
        else:
            break

    return age_freq


if __name__ == "__main__":
    datafile = read_file()
    datafile = fill_age(datafile)

    graph = population_pyramid(datafile)
    print(graph)
"""
# distribution of location embarked from
ax = sns.countplot(data=df, x="Embarked")
ax.bar_label(ax.containers[0])

plt.title("Embarked Distribution")
plt.xlabel("Location")
plt.ylabel("Count")
plt.show()

# distribution of sex
ax = sns.countplot(data=df, x="Sex")
ax.bar_label(ax.containers[0])

plt.title("Passenger Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# creates a correlation matrix between the specified variables
corr_data = df[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
corr_data = corr_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# building and deploying random forest model
X = df.drop(['Survived'], axis=1)
y = df['Survived']  # setting target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)  # train test split

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

print(X_train.head())
print(X_test.head())

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

# Save the trained model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest, file)

# Load the trained model from the file
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model for predictions
loaded_model_accuracy = loaded_model.score(X_test, y_test)
print("Loaded Model Accuracy Score:", round(loaded_model_accuracy * 100, 2))

app = Flask(__name__)


# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the request form
    age = float(request.form['age'])
    siblings = int(request.form['siblings'])
    sex = request.form['sex']  # Assuming 'Gender' is a dropdown with 'Male' or 'Female'

    # Map sex to numerical values (0 for female, 1 for male)
    sex_mapping = {'Female': 0, 'Male': 1}
    sex_numeric = sex_mapping.get(sex)

    # Prepare the input features for prediction with column names
    input_features = pd.DataFrame([[age, 0, 0, siblings]], columns=['Age', 'SibSp', 'Sex_female', 'Sex_male'])

    if sex_numeric == 0:
        input_features['Sex_female'] = 1
    else:
        input_features['Sex_male'] = 1

    # Make prediction using the model
    prediction = loaded_model.predict(input_features)

    # Render prediction result in HTML template
    if prediction[0] == 1:
        result_text = 'Survived'
    else:
        result_text = 'Did not survive'

    return render_template('index.html', prediction_text=result_text)

"""


