# Image Classification with CNN Models

This repository showcases the development of convolutional neural network (CNN) models for image classification. The project covers various techniques to enhance model performance, including architecture changes, hyperparameter tuning, reducing the learning rate, and using data augmentation.

## Project Overview
The main goal of this project is to train CNN models to classify images into multiple categories. Various model architectures and training strategies are explored to improve accuracy, including:
1. **Architecture Changes**: Experimenting with different numbers of convolutional and fully connected layers.
2. **Learning Rate Adjustment**: Reducing the learning rate to stabilize training and achieve better convergence.
3. **Data Augmentation**: Applying transformations such as rotation, zoom, shear, and flips to increase the diversity of the training dataset.

## Models Implemented
Here is a summary of the models implemented from Model 1 to Model 14, detailing the changes and improvements made at each step:

1. **Model 1: Basic CNN Architecture**
   - Consists of two convolutional layers with max-pooling.
   - Uses a learning rate of 0.01.
   - Achieved around 47% training accuracy after 10 epochs.

2. **Model 2: Increased Filter Size**
   - Added more filters in the convolutional layers.
   - Achieved slightly better accuracy than Model 1 (around 50%).

3. **Model 3: Added Another Convolutional Layer**
   - Added a third convolutional layer to improve feature extraction.
   - Training accuracy improved to around 53%.

4. **Model 4: Dropout for Regularization**
   - Introduced dropout layers to prevent overfitting.
   - Achieved similar accuracy to Model 3 but showed more stable training behavior.

5. **Model 5: Increased Number of Epochs**
   - Trained for 20 epochs instead of 10 to improve accuracy.
   - Achieved approximately 56% accuracy after 20 epochs.

6. **Model 6: Batch Normalization Added**
   - Added batch normalization layers to improve model performance.
   - Reached a training accuracy of around 59%.

7. **Model 7: Learning Rate Adjustment**
   - Lowered the learning rate to 0.001 for more stable training.
   - Achieved around 61% training accuracy.

8. **Model 8: Further Dropout Adjustments**
   - Adjusted dropout rates for better regularization.
   - Improved training accuracy to around 62%.

9. **Model 9: Increased Number of Filters**
   - Increased the number of filters in each convolutional layer.
   - Achieved around 63% accuracy.

10. **Model 10: Additional Fully Connected Layer**
    - Added another fully connected layer to the network.
    - Training accuracy reached approximately 64%.

11. **Model 11: Optimizer Changed to Adam**
    - Changed the optimizer from SGD to Adam for better convergence.
    - Achieved training accuracy of around 65%.

12. **Model 12: Learning Rate Scheduling**
    - Implemented a learning rate schedule to gradually reduce the learning rate during training.
    - Training accuracy improved to approximately 66%.

13. **Model 13: Learning Rate Adjustment**
   - A relatively simple CNN architecture with three convolutional layers followed by max-pooling layers.
   - The learning rate is set to 0.001 using the SGD optimizer.
   - Achieved a training accuracy of approximately 63% after 10 epochs.

14. **Model 14: Data Augmentation**
    - Trained the CNN model using an augmented dataset.
    - Applied random transformations such as rotation, zoom, brightness, and contrast changes.
    - Achieved a training accuracy of approximately 53% after 10 epochs, with improvements in validation accuracy.

## Evaluation
After training, the model's performance on the validation set will be evaluated and plotted using matplotlib.

**Example Plots**
Training vs. Validation Accuracy
Training vs. Validation Loss

## Results
The results of training and validation are plotted to visualize model performance. Accuracy and loss trends for both training and validation datasets are shown for different models to highlight the improvements achieved with various techniques.

## Future Work
- Experimenting with more complex CNN architectures.
- Using transfer learning to leverage pre-trained models.
- Further hyperparameter tuning and fine-tuning.

## Contributing
Feel free to fork the repository and submit pull requests. Any improvements and suggestions are welcome!
