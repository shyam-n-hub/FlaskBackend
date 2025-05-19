import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the dataset
df = pd.read_csv('dataset.csv')

# Step 2: Convert categorical data to numbers
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['performance'] = df['performance'].map({'Poor': 0, 'Average': 1, 'Excellent': 2})

# Step 3: Split features and target
X = df[['gender', 'study_time', 'attendance', 'previous_marks', 'behavior_score', 'internet_usage']]
y = df['performance']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Save the model
joblib.dump(model, 'student_model.pkl')

print("✅ Model trained and saved as student_model.pkl")
