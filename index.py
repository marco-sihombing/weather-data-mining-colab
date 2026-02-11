import pandas as pd

from google.colab import files
uploaded = files.upload()

df = pd.read_csv(list(uploaded.keys())[0])

df.head()

df.info()
print("\nMissing values:")
print(df.isnull().sum())

print("\nWeather distribution:")
print(df['weather'].value_counts())

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Original drop date
df.drop(columns=['date'], inplace=True)

df.head()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])

df.head()

from sklearn.model_selection import train_test_split

X = df.drop('weather', axis=1)
y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Sample prediction
sample = pd.DataFrame([{
    'precipitation': 0.0,
    'temp_max': 12.8,
    'temp_min': 5.0,
    'wind': 4.7,
    'year': 2012,
    'month': 1,
    'day': 1
}])

prediction = model.predict(sample)
print("Predicted weather:", le.inverse_transform(prediction))