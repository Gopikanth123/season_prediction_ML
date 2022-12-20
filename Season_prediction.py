# Import necessary libraries
import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the data
data =pd.read_csv(r"C:\Users\gopik\OneDrive\Documents\hack.csv")
print(data)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(data['Season'],test_size=0.2)
print(X_train, X_test, y_train, y_test)
# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
score = model.score(X_test, y_test)
print("Model accuracy:", score)


#IMPORTS
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


#DATA HANDLING
data = pd.read_csv(r"C:\Users\gopik\OneDrive\Documents\hack.csv")
print(data.head())
print(data.describe())
X = data[["Tourist Place","Climate"]]
Y = data["Tourist Place"]
Y_ = data["Class"]


#DATA ANALYSIS
plt.scatter(X['Tourist Place'], Y, color='b')
plt.xlabel('Tourist Place')
plt.ylabel('Climate')
plt.show()


#OBSERVATIONS
# print("The highest number of houses has 5.5 to 6.5 rooms with the price in between 12 and 30.")

#INPUTS
inps = [[3,7]]

#LINEAR REGRESSION
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict(inps)
print("Predicted value (LR): ",pred[0])
print("Accuracy (LR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Tourist Place'], Y, color='b')
plt.plot(X['Tourist Place'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Tourist Place (LR)')
plt.ylabel('Season')
plt.show()

