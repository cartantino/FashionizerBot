import pickle
import pprint
import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

DATA_ABSOLUTE = 'data/'
MODEL_ABSOLUTE = 'classifier/'

# Get features extracted from resnet18 model into features_data
with open(DATA_ABSOLUTE+'features_resnet18.pickle', 'rb') as handle:
    features_data = pickle.load(handle)


filenames = features_data['filenames']
labels = features_data['labels']



features_neurali = features_data['neural_features']
# bow_features = features_data['bow_features']

#bow_features = features_data['features']
labels = features_data['labels']
#print(bow_features.shape)

## Train SVM Model on Neural Features (SUBCATEGORIES)
Xtrain, Xtest, ytrain, ytest = train_test_split(features_neurali, labels, test_size=0.2)

# Normalize features before training
scaler = StandardScaler()
X_train = scaler.fit_transform(Xtrain)
X_test = scaler.transform(Xtest)

with open(DATA_ABSOLUTE+'scaler_nn.pickle', 'wb') as handle:
        pickle.dump(scaler, handle)

start = time.time()

# params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#                     {'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

print('Start training at ' + str(start))

# svm_model = GridSearchCV(SVC(class_weight= "balanced", verbose=True), params_grid, cv=3)
# svm_model.fit(X_train, ytrain)

svm_model = SVC(kernel = "rbf", verbose=False, probability=True, class_weight= "balanced", C = 1000, gamma=0.001)
svm_model.fit(Xtrain, ytrain)

print('End training, running time: %.4f seconds' % (time.time()-start))

print('Saving model..')
with open(MODEL_ABSOLUTE + 'SVM_resnet18_neural_features_2.pickle', 'wb') as handle:
    pickle.dump(svm_model, handle)


preds = svm_model.predict(Xtest)

# print('Best score for training data:', svm_model.best_score_,"\n") 
# # View the best parameters for the model found using grid search
# print('Best C:',svm_model.best_estimator_.C,"\n") 
# print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
# print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

# final_model = svm_model.best_estimator_
# preds = final_model.predict(Xtest)

print('Classification Report')
print(classification_report(ytest, preds))

print('Confusion Matrix')
try:
    pprint(confusion_matrix(ytest, preds))
except:
    print(confusion_matrix(ytest, preds))
