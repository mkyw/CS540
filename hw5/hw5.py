import sys
# import matplotlib.pyplot as plt
import numpy as np

# This contains the first argument as string
sys.argv[1]

# Create an array containing all the csv data
arr = np.genfromtxt(fname=sys.argv[1], delimiter=',', names = ['year', 'days'], skip_header=1)

# Plot
# fig, ax = plt.subplots()
# ax.plot(arr['year'], arr['days'])
# ax.set(xlabel='Year', ylabel= 'Number of frozen days')
# plt.savefig("plot.jpg")

# Q3a
X = np.zeros(shape=[len(arr), 2], dtype=np.int64)
for i in range(len(arr)):
    # a = np.array([1,arr[i][0]], ndmin=2)
    X[i][0] = 1
    X[i][1] = arr[i][0]

print("Q3a:")
print(X)

# Q3b
Y = np.empty(shape=[len(arr), 1], dtype=np.int64)
for i in range(len(arr)):
    Y[i] = arr[i][1]
print_y = np.transpose(Y)
print("Q3b:")
print(print_y[0])

# Q3c
Z = np.dot(np.transpose(X), X)
print("Q3c:")
print(Z)

# Q3d
I = np.linalg.inv(Z)
print("Q3d:")
print(I)

# Q3e
PI = np.dot(I, np.transpose(X))
print("Q3e:")
print(PI)

# Q3f
hat_beta = np.dot(PI, Y)
print("Q3f:")
print_hat_beta = np.transpose(hat_beta)
print(print_hat_beta[0])

# Q4
x_test = 2021
y_test = hat_beta[0][0] + np.dot(hat_beta[1][0], x_test)
print("Q4: " + str(y_test))

# Q5
if (hat_beta[1] > 0):
    print("Q5a: >")
elif (hat_beta[1] < 0):
    print("Q5a: <")
else:
    print("Q5a: =")
print("Q5b: Since we are performing a linear regression and beta is the degree of change, a negative beta value signifies that the number of days the lake is frozen is decreasing.")

# Q6
x_not = (-hat_beta[0][0])/(hat_beta[1][0])
print("Q6a: " + str(x_not))
print("Q6b: While the data of frozen snow days is obviously decreasing, and a cursory look at the data indicates five winters of the lake frozen for 100 days or more whereas there was only 3 winters of less than 100 frozen days in the first 20 years of data, the model is predicting ahead 400 years based on less than 200 years of data. I believe the sample size is not big enough for the model to predict that far into the future, and thus I don't find the results compelling.")
