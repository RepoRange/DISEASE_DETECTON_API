# Part of SFA

## Plant Disease Prediction Model
This section details the image classification model used within the Smart Farming Assistant (SFA) project to predict plant diseases based on leaf images.

### 1. Purpose: What does it do?

The primary goal of this model is to automatically identify the type of disease present on a plant leaf (or determine if it's healthy) by analyzing an image of that leaf. By providing farmers with a quick and accessible way to diagnose potential issues, it aids in early detection and treatment, potentially improving crop yield and reducing losses.

The model is trained to recognize **38 different classes**, covering various diseases across several plant types (like Apple, Corn, Grape, Tomato, etc.) as well as healthy examples.

### 2. Why Convolutional Neural Networks (CNNs)?

Image recognition is a complex task for traditional programming. How do you *tell* a computer what "Apple Scab" looks like using rules? It's incredibly difficult. This is where CNNs excel.

*   **Automatic Feature Learning:** CNNs are a type of deep learning model specifically designed for image data. They automatically learn to identify important visual features directly from the images during training.
*   **Hierarchical Learning:** They learn simple features first (like edges and colors) in early layers, and then combine these in later layers to recognize more complex patterns (like textures, spots, or specific shapes characteristic of a disease).
*   **Spatial Awareness:** CNNs use convolutional layers that scan across the image, preserving the spatial relationship between pixels, which is crucial for understanding shapes and objects.
*   **Efficiency:** Techniques like Pooling layers help reduce the size of the data while retaining the most important information, making the model more efficient.

In short, CNNs are the standard and most effective tool for image classification tasks like this one because they "learn to see" patterns relevant to the task without needing humans to define those patterns explicitly.

### 3. Dataset Used

*   **Source:** The model was trained on the popular **PlantVillage dataset**, obtained from Kaggle. (Dataset ID: `abdallahalidev/plantvillage-dataset`).
*   **Content:** The specific subset used consists of **color images** of plant leaves.
*   **Classes:** The dataset contains images categorized into **38 distinct classes** (different plant/disease combinations and healthy leaves).
*   **Size:** Over 54,000 color images were used for training and validation.

### 4. Model Architecture: How is it built?

The model uses a **Sequential CNN architecture**, built using TensorFlow/Keras. This means layers are stacked one after another. The specific structure is relatively simple but effective for this task:

1.  **Input Layer:** Accepts RGB color images resized to **224x224 pixels**.
2.  **Convolutional Layer 1:**
    *   Uses 32 filters (kernels) of size 3x3.
    *   Scans the input image to detect basic features (edges, corners, color patches).
    *   Uses ReLU (Rectified Linear Unit) activation function to introduce non-linearity.
3.  **Max Pooling Layer 1:**
    *   Downsamples the feature map using a 2x2 window.
    *   Reduces dimensionality and makes the model more robust to small variations in feature positions.
4.  **Convolutional Layer 2:**
    *   Uses 64 filters (more filters to learn more complex combinations of features from the previous layer) of size 3x3.
    *   Again, uses ReLU activation.
5.  **Max Pooling Layer 2:**
    *   Further downsamples the feature map (2x2 window).
6.  **Flatten Layer:**
    *   Converts the 2D feature maps resulting from the convolutional/pooling layers into a single 1D vector. This prepares the data for the final classification layers.
7.  **Dense Layer (Hidden):**
    *   A fully connected layer with 256 neurons.
    *   Uses ReLU activation. It learns higher-level combinations of the features extracted by the convolutional layers.
8.  **Output Layer (Dense):**
    *   A fully connected layer with **38 neurons** (one for each possible plant disease class).
    *   Uses **Softmax activation**. This function outputs a probability distribution across all 38 classes, indicating the model's confidence that the input image belongs to each class. The class with the highest probability is the final prediction.

*(Based on `model.summary()`, this architecture has approximately 47.8 million trainable parameters, mostly in the first Dense layer.)*

### 5. Training Process: How did it learn?

1.  **Data Preparation:**
    *   The PlantVillage dataset (color images) was used.
    *   Images were automatically loaded and labelled using Keras' `ImageDataGenerator` and `flow_from_directory`.
    *   Images were **resized to 224x224 pixels** to match the model's input layer.
    *   Pixel values were **rescaled** from the original 0-255 range to a 0-1 range (by dividing by 255.0), which helps the model learn more effectively.
    *   The data was split into **80% for training** and **20% for validation**.
2.  **Compilation:**
    *   **Optimizer:** The 'adam' optimizer was used, which is a common and effective algorithm for adjusting the model's internal parameters (weights) during training.
    *   **Loss Function:** 'categorical_crossentropy' was chosen as the loss function. This is standard for multi-class classification problems where each image belongs to only one class. It measures how far the model's predicted probabilities are from the actual correct class.
    *   **Metric:** 'accuracy' was monitored during training to see the percentage of images correctly classified.
3.  **Training Loop:**
    *   The model was trained for **5 epochs** (meaning it processed the entire training dataset 5 times).
    *   Training was done in **batches of 32 images** at a time.
    *   After each epoch, the model's performance was evaluated on the separate validation set to monitor its generalization ability (how well it performs on unseen data).

### 6. Evaluation: How well does it perform?

After 5 epochs of training, the model achieved a **validation accuracy of approximately 88.28%**. This means it correctly identified the disease (or healthy status) for about 88 out of every 100 images in the unseen validation set. The training and validation accuracy/loss curves (plotted in the notebook) help visualize how the model learned over the epochs.

### 7. How it Works in SFA (Prediction Pipeline)

When a user uploads a leaf image to the SFA API:

1.  **Preprocessing:** The input image is loaded, resized to 224x224 pixels, and its pixel values are scaled to the 0-1 range, exactly like the training data.
2.  **Prediction:** This preprocessed image is fed into the loaded, pre-trained model (`plant_disease_prediction_model.h5`).
3.  **Output:** The model outputs 38 probability scores (thanks to the Softmax layer).
4.  **Classification:** The class (disease name) corresponding to the highest probability score is identified using the `class_indices.json` mapping file.
5.  **Result:** This predicted class name is returned to the user via the API.

This model forms the core of the disease detection feature within the Smart Farming Assistant.



### Kaggle Dataset Link:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset


### Disease_Detection_API_Download_Link:
https://drive.google.com/drive/folders/1H-ix3SiUaeoVZC2uTFaMVuCo7zaZlh_G?usp=drive_link

