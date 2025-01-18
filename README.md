# Sign Language Detection using CNN

This project implements a Convolutional Neural Network (CNN) model for detecting and recognizing sign language gestures. The model is trained to classify gestures into specific categories and can be used for real-time sign language detection.

## Project Structure

- **trainmodel.ipynb**: Contains the implementation of the CNN model, training pipeline, and evaluation metrics. The model runs for 10 epochs.
- **realtimedetection.py**: A script to perform real-time detection using the trained model.
- **signlanguageASLDataset.json**: The JSON file containing the model architecture and weights.
- **README.md**: Provides an overview of the project.

## CNN Model Architecture
The CNN model includes the following layers:

```python
model = Sequential()
# Convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# Fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
```

## Training
- The model is trained for **10 epochs** using a dataset of sign language gestures.
- The training process is implemented in `trainmodel.ipynb`.

## Real-Time Detection
- The trained model is exported as a JSON file.
- The `realtimedetection.py` script loads the model from the JSON file and uses it to perform real-time detection.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/abhijeetvr12/Sign-Language-Detection-Using-CNN/
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   - Open `trainmodel.ipynb`.
   - Run all cells to train the CNN model.
   - The trained model will be saved as `model.json`.

4. Perform real-time detection:
   - Run the `realtimedetection.py` script:
     ```bash
     python realtimedetection.py
     ```

## Dataset
- The dataset consists of grayscale images of sign language gestures with a resolution of 48x48 pixels.
- Images are preprocessed to fit the input shape `(48, 48, 1)`.

## Features
- **Robust Architecture**: The CNN model is designed with multiple convolutional and fully connected layers to handle complex gestures.
- **Real-Time Detection**: The model supports real-time gesture recognition via webcam.

## Future Work
- Expand the dataset to include more gestures.
- Optimize the model for faster real-time detection.
- Integrate additional preprocessing steps to improve accuracy.

## Contributors
Abhijeet Kumar

## License
This project is licensed under the MIT License.
