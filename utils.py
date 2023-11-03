from sklearn import datasets, metrics, svm, tree
import itertools
from sklearn.model_selection import train_test_split
from joblib import dump


def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y
#metric = metrics.accuracy_score

def train_model(x, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC
    if model_type == "tree":
        clf = tree.DecisionTreeClassifier

    model = clf(**model_params)
    # pdb.set_trace()
    model.fit(x, y)
    return model 

def data_preprocess(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data 


#function for getting the train test and dev split
def train_dev_test_split(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= test_size,shuffle=False
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size = dev_size/(1-test_size), shuffle = False
    )

    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
    predicted_val = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_pred=predicted_val, y_true=y_test)
    return accuracy

def hparams_combinations(gamma_list, c_list):
    return list(itertools.product(gamma_list, c_list))

def tune_hparams(X_train, y_train, X_dev, y_dev, combinations, model_type = 'svm'):
    best_accuracy = -1
    best_model=None
    best_hparams = None
    best_model_path = ""

    for param in combinations:
        cur_model = train_model(X_train,y_train,{'gamma':param[0],'C':param[1]},model_type='svm')
        cur_accuracy = predict_and_eval(cur_model,X_dev,y_dev)
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_hparams=param
            best_model = cur_model
            best_model_path = "./models/{}_".format(model_type) + "_".join(["{}:{}".format(k,v) for k,v in param.items()]) + ".joblib"
    dump(best_model, best_model_path)

        
    return best_hparams, best_model_path, best_accuracy
    
