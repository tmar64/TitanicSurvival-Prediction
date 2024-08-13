from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pprint

# global variables for age ranges
dict_keys = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34',
             '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-80']


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


# returns frequency of males in specific age ranges
def age_frequency_male(df):
    # create df of m/f and age
    sex_age = df[['Sex', 'Age']]

    # creates df of m and age
    male_age = sex_age[sex_age.Sex == 'male']

    # sort df in ascending order, by age
    male_age = male_age.sort_values(by=['Age'], ascending=True)

    # create series of age and their frequency
    male_age_frequency = male_age['Age'].value_counts(sort=False)

    # create dict of age ranges set to 0
    male_dict = dict.fromkeys(dict_keys, 0)

    # iterate through panda series, if age is in specified range, increment in dict
    for age, freq in male_age_frequency.items():
        if age <= 4:
            male_dict['0-4'] += freq
        elif age <= 9:
            male_dict['5-9'] += freq
        elif age <= 14:
            male_dict['10-14'] += freq
        elif age <= 19:
            male_dict['15-19'] += freq
        elif age <= 24:
            male_dict['20-24'] += freq
        elif age <= 29:
            male_dict['25-29'] += freq
        elif age <= 34:
            male_dict['30-34'] += freq
        elif age <= 39:
            male_dict['35-39'] += freq
        elif age <= 44:
            male_dict['40-44'] += freq
        elif age <= 49:
            male_dict['45-49'] += freq
        elif age <= 54:
            male_dict['50-54'] += freq
        elif age <= 59:
            male_dict['55-59'] += freq
        elif age <= 64:
            male_dict['60-64'] += freq
        elif age <= 69:
            male_dict['65-69'] += freq
        elif age <= 74:
            male_dict['70-74'] += freq
        elif age <= 80:
            male_dict['75-80'] += freq
        else:
            break

    # return dictionary in clean format using pprint
    return pprint.pprint(male_dict)


# returns frequency of females in specific age ranges
def age_frequency_female(df):
    # create df of m/f and age
    sex_age = df[['Sex', 'Age']]

    # creates df of f and age
    female_age = sex_age[sex_age.Sex == 'female']

    # sort df in ascending order, by age
    female_age = female_age.sort_values(by=['Age'], ascending=True)

    # create series of age and their frequency
    female_age_frequency = female_age['Age'].value_counts(sort=False)

    # create dict of age ranges set to 0
    female_dict = dict.fromkeys(dict_keys, 0)

    # iterate through panda series, if age is in specified range, increment in dict
    for age, freq in female_age_frequency.items():
        if age <= 4:
            female_dict['0-4'] += freq
        elif age <= 9:
            female_dict['5-9'] += freq
        elif age <= 14:
            female_dict['10-14'] += freq
        elif age <= 19:
            female_dict['15-19'] += freq
        elif age <= 24:
            female_dict['20-24'] += freq
        elif age <= 29:
            female_dict['25-29'] += freq
        elif age <= 34:
            female_dict['30-34'] += freq
        elif age <= 39:
            female_dict['35-39'] += freq
        elif age <= 44:
            female_dict['40-44'] += freq
        elif age <= 49:
            female_dict['45-49'] += freq
        elif age <= 54:
            female_dict['50-54'] += freq
        elif age <= 59:
            female_dict['55-59'] += freq
        elif age <= 64:
            female_dict['60-64'] += freq
        elif age <= 69:
            female_dict['65-69'] += freq
        elif age <= 74:
            female_dict['70-74'] += freq
        elif age <= 80:
            female_dict['75-80'] += freq
        else:
            break

    # return dictionary in clean format using pprint
    return pprint.pprint(female_dict)


# return population pyramid
def population_pyramid():

    # pandas df created using the values from the age_frequency functions
    pop_pyramid = pd.DataFrame(
        {'Age': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                 '50-54', '55-59', '60-64', '65-69', '70-74', '75-80'],
         'Male': [-36, -12, -10, -65, -90, -92, -75, -62, -35, -34, -28, -15, -11, -5, -6, -1],
         'Female': [20, 14, 8, 46, 48, 36, 37, 37, 26, 16, 13, 6, 6, 1, 0, 0]})

    # set age range for the Y axis
    age_ranges = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                  '50-54', '55-59', '60-64', '65-69', '70-74', '75-80']

    # create the population pyramid
    bar_plot = sns.barplot(x='Male', y='Age', data=pop_pyramid, order=age_ranges)
    bar_plot = sns.barplot(x='Female', y='Age', color='Pink', data=pop_pyramid, order=age_ranges)
    bar_plot.set(xlabel='Population', ylabel='Age', title='Age/Gender Distribution on the Titanic')

    bar_plot.set_xticklabels([int(max(x, -x)) for x in bar_plot.get_xticks()])

    bar_plot.invert_yaxis()

    return plt.show()


if __name__ == "__main__":
    # load and create df
    datafile = read_file()

    # fill missing age with ffill method
    datafile = fill_age(datafile)

    # print age distribution of males and female
    print("Male Age Distribution")
    print(age_frequency_male(datafile), '\n')
    print("Female Age Distribution")
    print("female", age_frequency_female(datafile))

    # call population pyramid
    population_pyramid()


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
