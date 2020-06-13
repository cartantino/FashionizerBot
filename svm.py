import pickle
import pprint
import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_ABSOLUTE = 'data/'
MODEL_ABSOLUTE = 'classifier/'

# Get features extracted from resnet18 model into features_data
with open(DATA_ABSOLUTE+'features_resnet18.pickle', 'rb') as handle:
    features_data = pickle.load(handle)


filenames = features_data['filenames']
with open(DATA_ABSOLUTE+'filenames.pickle', 'wb') as handle:
        pickle.dump(filenames, handle)

labels = features_data['labels']
with open(DATA_ABSOLUTE+'labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle)


features_neurali = features_data['neural_features']
bow_features = features_data['bow_features']


## Train SVM Model on Neural Features (SUBCATEGORIES)
Xtrain, Xtest, ytrain, ytest = train_test_split(features_neurali, labels, test_size=0.2)

# Normalize features before training
scaler = StandardScaler()
X_train = scaler.fit_transform(Xtrain)
X_test = scaler.transform(Xtest)

with open(DATA_ABSOLUTE+'scaler.pickle', 'wb') as handle:
        pickle.dump(scaler, handle)

start = time.time()

print('Start training at ' + str(start))

svm_model = SVC(kernel = "rbf", verbose=True, probability=True, class_weight= "balanced")
svm_model.fit(Xtrain, ytrain)

print('End training, running time: %.4f seconds' % (time.time()-start))

print('Saving model..')
with open(MODEL_ABSOLUTE + 'SVM_resnet18_neural_features.pickle', 'wb') as handle:
    pickle.dump(svm_model, handle)


preds = svm_model.predict(Xtest)


print('Classification Report')
print(classification_report(ytest, preds))

print('Confusion Matrix')
try:
    pprint(confusion_matrix(ytest, preds))
except:
    print(confusion_matrix)

