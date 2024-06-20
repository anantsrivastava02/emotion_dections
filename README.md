# Emotion Detection Project

This repository contains code for an emotion detection project, which achieves an accuracy of 66%. The project includes four different models, each utilizing a different approach for emotion detection. Below, you will find a detailed description of each model and the corresponding evaluation results.

## Dataset

The project utilizes the "fer2013" dataset from Kaggle, provided by the user "msambare". This dataset consists of facial images labeled with different emotions.

## Model 1: CNN Model from Scratch

This model is built using a Convolutional Neural Network (CNN) architecture from scratch. The layers used in this model are as follows:

```python
model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer="glorot_uniform", padding='same', input_shape=(img_width, img_height, 1)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Conv2D(256, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

Evaluation results for this model include:

- AUC_ROC Curve: The area under the Receiver Operating Characteristic (ROC) curve is plotted to measure the model's ability to distinguish between different emotions.
- Confusion Matrix: A confusion matrix is created to visualize the model's performance across different emotions.
- Classification Report: A classification report is generated to provide a detailed analysis of the model's performance, including metrics such as precision, recall, and F1-score for each emotion class.

## Model 2: CNN Model from Scratch with Augmentation Learning

Similar to Model 1, this model is also built using a CNN architecture from scratch. In addition, augmentation learning techniques are applied to improve the model's performance. The evaluation results for this model include:

- Confusion Matrix: A confusion matrix is plotted to visualize the model's performance across different emotions.
- Classification Report: A classification report is generated to provide a detailed analysis of the model's performance.
- Early Stop Callbacks: Early stop callbacks are implemented to monitor the model's training progress and prevent overfitting.

## Model 3: Transfer Learning via VGG16

This model utilizes transfer learning with the VGG16 pre-trained model. The last three layers of the VGG16 model are made trainable, while the rest of the layers are frozen. The evaluation results for this model include:

- Confusion Matrix: A confusion matrix is plotted to visualize the model's performance across different emotions.
- AUC_ROC Curve: The area under the ROC curve is plotted to measure the model's ability to distinguish between different emotions.

## Model 4: Transfer Learning via ResNet50

In this model, transfer learning is applied using the pre-trained ResNet50 model. The pre-trained weights are used to initialize the model, and the last layer is modified for emotion detection. The evaluation results for this model include:

- Confusion Matrix: A confusion matrix is plotted to visualize the model's performance across different emotions.
- Accuracy: The accuracy of the model is calculated and reported.

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository to your local machine: `git clone https://github.com/your-username/emotion-detection.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the "fer2013" dataset from Kaggle and place it in the appropriate directory.
4. Run the models: Refer to the documentation in the respective model directories for instructions on how to run each model.
5. View the evaluation results: The evaluation results, including the plotted graphs and generated reports, can be found in the output directories of each model.

## Requirements

The following dependencies are required to run the models:

- Python 3.6 or above
- TensorFlow 2.0 or above
- Keras 2.3.1 or above
- Matplotlib 3.2.1 or above
- Scikit-learn 0.23.1 or above

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

We would like to acknowledge the following resources that were instrumental in the development of this project:

- [msambare/fer2013 Dataset](https://www.kaggle.com/msambare/fer2013) - The dataset used for training and evaluation.
- [TensorFlow Documentation](https://www.tensorflow.org) - The official TensorFlow documentation, which provided valuable insights into building and training deep learning models.

## Contact

For any inquiries or further information about this project, please contact [your-email@example.com](mailto:your-email@example.com).

Happy emotion detecting!
