##################      Softmax test      ############################
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

digits = load_digits()

X = digits['data']
y = digits['target']

normalizer = StandardScaler()
X = normalizer.fit_transform(X)

print('X shape :', X.shape)
print('Y shape :', y.shape)

# split train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# set architecture
batch_size = 256
alpha = 0.5
hidden_units = [100, 100, 50]
hidden_func='tanh'
output_func='softmax'


NN_clf = AdamANN_clf(alpha=alpha, hidden_units=hidden_units, hidden_func=hidden_func, \
                          batch_size=batch_size, output_func=output_func, 
                          epoch=200, learning_rate=0.01, learn_decay=0.0001, grad_check=False)

NN_clf.fit(X_train, y_train)
# compute train accuracy
y_train_pred = NN_clf.predict(X_train)
train_accuracy = accuracy_score(y_train_pred, y_train)

# compute test accuracy
y_test_pred = NN_clf.predict(X_test)
test_accuracy = accuracy_score(y_test_pred, y_test)

print('Train accuracy :', train_accuracy)
print('Test accuracy :', test_accuracy)