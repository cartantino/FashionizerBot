import pickle
import cv2

# Get features extracted from resnet18 model into features_data
with open('mask/current_mask.pickle', 'rb') as handle:
    current_mask = pickle.load(handle)

with open('mask/result.pickle', 'rb') as handle:
    result = pickle.load(handle)

print(result)
result[current_mask != 1] = 255

cv2.imshow("bbox", result)
cv2.waitKey(0)