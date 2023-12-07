import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For older scikit-learn versions

# Load your images and preprocess them
# Replace "path/to/your/images" with the actual path to your images
images_up = [cv2.imread("/home/vignesh/Pictures/up_arrow/up_{}.png".format(i)) for i in range(1, 18)]
images_down = [cv2.imread("/home/vignesh/Pictures/down_arrow/down_{}.png".format(i)) for i in range(1, 18)]
images_right = [cv2.imread("/home/vignesh/Pictures/left_arrow/left_{}.png".format(i)) for i in range(1, 18)]
images_left = [cv2.imread("/home/vignesh/Pictures/right_arrow/right_{}.png".format(i)) for i in range(1, 18)]

# Assuming all images have the same size, adjust this based on your dataset
image_size = (64, 64)

# Resize images and convert to grayscale
images_up = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), image_size) for img in images_up]
images_down = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), image_size) for img in images_down]
images_right = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), image_size) for img in images_right]
images_left = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), image_size) for img in images_left]

# Create feature matrix X and label vector y
X = np.vstack([
    img.flatten() for img_list in [images_up, images_down, images_right, images_left]
    for img in img_list
])
y = np.array(['up'] * 17 + ['down'] * 17 + ['right'] * 17 + ['left'] * 17)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = svm.SVC(kernel='rbf')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained SVM model to a file
model_filename = "svm_model.pkl"
joblib.dump(svm_classifier, model_filename)
