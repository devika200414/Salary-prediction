import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("salary_data.csv")

# Input (experience) and output (salary)
X = data[["experience"]]
y = data["salary"]

# Create and train the AI model
model = LinearRegression()
model.fit(X, y)

# Take user input
years = float(input("Enter years of experience: "))

# Prepare input for prediction (with column name)
input_data = pd.DataFrame([[years]], columns=["experience"])

# Predict salary
prediction = model.predict(input_data)

# Print the result
print("Predicted Salary:", int(prediction[0]))

# --------- GRAPH PART ---------
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")

# Predicted point
plt.scatter(years, prediction, color="red", label="Predicted Salary")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary Prediction")
plt.savefig("salary_prediction.png",dpi=300,bbox_inches="tight")
plt.legend()

plt.show()