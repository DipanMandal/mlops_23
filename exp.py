"""t@github.com:DipanMandal/mlops_23.git"""
"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.


"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import train_dev_test_split, predict_and_eval, data_preprocess, tune_hparams, hparams_combinations
from sklearn.metrics import accuracy_score, confusion_matrix

digits = datasets.load_digits()
X = digits.images
y = digits.target

#print(f"size of the images: {digits.data.shape}")
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
#--------------------------------------------------
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
#-------------------------------------------------
# Create a classifier: a support vector classifier
#===========================================================================================================
#clf = svm.SVC(gamma=0.001)

# #we are taking 30% for test set and 20% for the dev set
# X_train, X_test, y_train, y_test, dev_train, dev_test = train_dev_test_split(data, digits.target, 0.3, 0.3)

# #training the model in the cross validation set
# clf.fit(X_train, y_train)    
# predicted_dev = predict_and_eval(clf, X_test, dev_test)
# print("Cross-validation data prediction: ",predicted_dev)
#===========================================================================================================
# clf.fit(X_train, y_train)
# predicted = predict_and_eval(clf, X_test, y_test)
# print("Test data prediction: ", predicted)

###############################################################################
# # Below we visualize the first 4 test samples and show their predicted
# # digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted_dev):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# ###############################################################################
# # :func:`~sklearn.metrics.classification_report` builds a text report showing
# # the main classification metrics.

# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted_dev)}\n"
# )

# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_dev)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

###############################################################################
# # If the results from evaluating a classifier are stored in the form of a
# # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# # as follows:


# # The ground truth and predicted lists
# y_true = []
# y_pred = []
# cm = disp.confusion_matrix

# # For each cell in the confusion matrix, add the corresponding ground truths
# # and predictions to the lists
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]

# print(
#     "Classification report rebuilt from confusion matrix:\n"
#     f"{metrics.classification_report(y_true, y_pred)}\n"
# )

#=================================================================================================================
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

combinations = hparams_combinations(gamma_list,c_list)

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test in test_sizes:
    for dev in dev_sizes:
        train_size = 1 - (test+dev)
        X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(X, y, test_size=test, dev_size=dev)

        X_train = data_preprocess(X_train)
        X_dev = data_preprocess(X_dev)
        X_test = data_preprocess(X_test)

        #print(f"for train size: {train_size} test size:{test}, dev size:{dev} :: train data size : {len(X_train)} test data size: {len(X_test)} dev data size: {len(X_dev)}")
        
        best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, combinations)
        
        # print(f"test_size={test} dev_size={dev} train_size={train_size} train_acc={best_accuracy:.2f} dev_acc={best_accuracy:.2f} test_acc={best_accuracy:.2f}")

print(f"Best Hyperparameters: ( gamma : {best_hparams[0]} , C : {best_hparams[1]} )")

clf = svm.SVC(gamma=best_hparams[0], C= best_hparams[1])
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)

candidate_model = DecisionTreeClassifier()
candidate_model.fit(X_train, y_train)
candidate_predict = candidate_model.predict(X_test)

clf_accuracy = accuracy_score(y_test, clf_predict)
candidate_accuracy = accuracy_score(y_test, candidate_predict)

print(f"production model accuracy = {clf_accuracy}")
print(f"candidate model accuracy = {candidate_accuracy}")

confusion_matrix_clf = confusion_matrix(y_test, clf_predict)
confusion_matrix_candidate = confusion_matrix(y_test, candidate_predict)

print(f"production model confusion matrix:\n {confusion_matrix_clf}")
print(f"candidate model confusion matrix:\n {confusion_matrix_candidate}")