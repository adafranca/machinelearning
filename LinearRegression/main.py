import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

''' A1, the percentage price deflation;
A2, the GNP in millions of dollars;
A3, the number of unemployed in thousands;
A4, the number of people employed by the military;
A5, the number of people over 14;
A6, the year,
B,  the number of people employed.
 '''

data = pd.read_csv("dataset.txt", sep="  ", header=None)
data.columns = ["A1", "A2", "A3", "A4", "A5", "A6", "B"]


''' Verify correlation between X and Y '''

Y = data['B']

plt.subplot(1, 3, 1)
plt.scatter(data['A1'], Y)

plt.subplot(1, 3, 2)
plt.scatter(data['A2'], Y)

plt.subplot(1, 3, 3)
plt.scatter(data['A3'], Y)
plt.show()

plt.subplot(1, 3, 1)
plt.scatter(data['A4'], Y)

plt.subplot(1, 3, 2)
plt.scatter(data['A5'], Y)

plt.subplot(1, 3, 3)
plt.scatter(data['A6'], Y)
plt.show()

''' Split Data '''

x_train, x_test, y_train, y_test  = train_test_split(data[['A1', 'A2', 'A3', 'A4', 'A5']], data[['B']])


''' Building Model '''

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)