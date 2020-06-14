import pickle
from sklearn.neighbors import KDTree
import numpy as np

DATA_ABSOLUTE = 'data/'
MODEL_ABSOLUTE = 'classifier/'
KDTREE_ABSOLUTE = 'kdtree_model/'


## EVALUATE KD-TREE WITH NEURAL FEATURES (RESNET18)

# Get features extracted from resnet18 model into features_data
with open(DATA_ABSOLUTE+'features_resnet18.pickle', 'rb') as handle:
    features_data = pickle.load(handle)

# Get KdTree model
with open(KDTREE_ABSOLUTE+'KDTree_neuralFeatures.pickle', 'rb') as handle:
    kdTree = pickle.load(handle)

filenames = features_data['filenames']
labels = features_data['labels']
neural_features = features_data['neural_features']

sizes = [1, 3, 5, 10]


for k_size in sizes:
    number_predictions = []
    print('****************************')
    print("Evaluating accuracy for size =  " + str(k_size))

    for filename, label, neural_feature in zip(filenames, labels, neural_features):
        dist, indexes = kdTree.query([neural_feature], k = k_size + 1) # return k images per image (+1 because the first retrieved image is the same image)
        #print("Dist = " + str(dist))
        #print("Indexes = " + str(indexes))
        retrieval_label = [labels[index] for index in indexes[0][1:]]
        #print("Retrieved labels = " + str(retrieval_label))
        correct_predicted = retrieval_label.count(label)
        #print("Predicted right = " + str(correct_predicted))
        number_predictions.append(correct_predicted)

    pikle_name = 'number_predictions_' + str(k_size)
    with open(DATA_ABSOLUTE + pikle_name + '.pickle', 'wb') as handle:
            pickle.dump(number_predictions, handle)


    mean_acc = np.mean(np.array(number_predictions)/k_size)
    print('****************************')
    print("Accuracy (mean_acc) for k = " + str(k_size) + " is " + str(mean_acc))