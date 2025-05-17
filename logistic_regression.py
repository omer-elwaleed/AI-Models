import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

print(df.head())

x = df[['StudyHours']]
y = df['Pass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


study_hours_range = np.linspace(x.min(), x.max(), 100)
y_proba = model.predict_proba(study_hours_range.reshape(-1, 1))[:, 1]

plt.scatter(x_test, y_test, color='blue', label='Actual Data')
plt.plot(study_hours_range, y_proba, color='red', label='Logistic Regression Curve')

plt.xlabel('Study Hours')
plt.ylabel('Probability of passing')
plt.title('Logistic Regression: Study Hours vs Probability of Passing')
plt.legend()

plt.show()