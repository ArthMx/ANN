##################      Softmax test      ############################
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

digits = load_digits()

X = digits['data']/16 # normalize the values between 0 and 1
y = digits['target']

print('X shape :', X.shape)
print('Y shape :', y.shape)

# split train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# set architecture
hidden_units = [10, 5]
hidden_func='tanh'
output_func='softmax'


NN_clf = ANN_clf(hidden_units, hidden_func, output_func, epoch=10000, learning_rate=0.1)

NN_clf.fit(X_train, y_train)
# compute train accuracy
y_train_pred = NN_clf.predict(X_train)
train_accuracy = accuracy_score(y_train_pred, y_train)

# compute test accuracy
y_test_pred = NN_clf.predict(X_test)
test_accuracy = accuracy_score(y_test_pred, y_test)

print('Train accuracy :', train_accuracy)
print('Test accuracy :', test_accuracy)