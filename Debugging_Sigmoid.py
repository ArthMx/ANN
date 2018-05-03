##################      Sigmoid test      ############################
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

digits = load_digits()

X = digits['data']/16 # normalize the values between 0 and 1
Y = digits['target']
Y = (Y == 1)*1  # target value =1 for digit 1, and 0 for all other digits

print('X shape :', X.shape)
print('Y shape :', Y.shape)

# split train/test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# set architecture
hidden_units = [20,10]
hidden_func='tanh'
output_func='sigmoid'

parameters = NN_model(X_train, Y_train, hidden_units, hidden_func, output_func, \
             epoch=10000, learning_rate=0.5, verbose=True)

# compute train accuracy
Y_train_pred = MakePrediction(X_train, parameters, hidden_func, output_func)
train_accuracy = accuracy_score(Y_train_pred, Y_train)

# compute test accuracy
Y_test_pred = MakePrediction(X_test, parameters, hidden_func, output_func)
test_accuracy = accuracy_score(Y_test_pred, Y_test)

print('Train accuracy :', train_accuracy)
print('Test accuracy :', test_accuracy)