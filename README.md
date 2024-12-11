# Fall-Detection (Personal Project)

Current Version: 2.0

Dataset: Harvard Dataverse's Fall Vision Dataset (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK)

Approach (V2.0): Sequence Modeling using a combination of CNN and LSTM
                  1) Model using ConvLSTM2D()
                  2) LRCN: Convolution and LSTM layers in a single model

## Timeline:
		
<img width="892" alt="Screenshot 2024-12-11 at 2 39 51 AM" src="https://github.com/user-attachments/assets/7f4be7a4-df78-4604-b8fe-c3cae78c7871">


## Version 1.0:

### ConvLSTM V1.0:

![image](https://github.com/user-attachments/assets/c8843f1d-190d-4ef3-8be0-c1853b317311)

#### Inferences:

1. High training accuracy: 0.95. indicates that the model is learning the patterns in the training data effectively.
2. Validation accuracy initially increases but plateaus and fluctuates around ~0.75 after a certain point. This could be due to overfitting, where the model starts to memorize the training data instead of generalizing well to unseen data.
	• The gap between the training and validation accuracy widens as training progresses. This is a typical sign of overfitting.
	• The early stopping mechanism might not have fully mitigated overfitting.

3. The training loss decreases consistently, which indicates that the model is fitting the training data well.
4. The validation loss initially decreases, then starts fluctuating and remains higher than the training loss. This supports the observation of overfitting from the accuracy graph.
  • The divergence between training and validation loss indicates overfitting.

#### Recommendations:

1. Data Augmentation:
	• Random cropping or resizing frames.
	• Applying random Cropping, rotations, flipping frames, or brightness adjustments to frames.
2. Regularization:
	• Increase dropout rates in the layers (currently set to 0.2).
	• Use L2 regularization (weight decay) in the Dense layers.
3. Batch Size Adjustment:
	• Experiment with larger batch sizes (e.g., 8 or 16) to improve generalization.
4. Learning Rate Adjustment:
	• Reduce the learning rate gradually using a learning rate scheduler or use a smaller initial learning rate.
5. Early Stopping:
	• Use stricter early stopping criteria, for example: patience=5.
6. Model Architecture:
	• Simplify the architecture by reducing the number of ConvLSTM layers or filters. 
	• The current architecture might be too complex for the dataset size.
7. Validation Strategy:
	• Use k-fold cross-validation to better evaluate the generalization capability of the model.
8. Sequence Length:
	• Currently, SEQUENCE_LENGTH = 3 means the model sees only three frames per video. 
	• This might result in a loss of temporal information, as the videos are 2-5 seconds long. 
	• Increasing SEQUENCE_LENGTH (e.g., to 10 or 20) allows the model to capture more temporal dynamics.
9. Frame Selection Strategy:
	• Instead of uniformly sampling frames, consider random sampling or stratified sampling to introduce variability:
		○ Randomize the frame selection process in frames_extraction
10. Normalize Labels:
	• Balance the dataset by oversampling the minority class (fall) or undersampling the majority class (no_fall).
11. Use a Validation Split:
	• Ensure that the validation set is representative of the dataset to better estimate generalization. 
	• Instead of directly splitting the data, stratify based on labels.
12. Feature Scaling:
	• Your current normalization (frame / 255) scales the pixel values between 0 and 1. 
	• This is good, but you can explore other normalization techniques (e.g., mean subtraction) to improve generalization.
13. Reduce Video Resolution:
	• Reduce IMAGE_HEIGHT and IMAGE_WIDTH further (e.g., 32x32) to decrease the model's complexity. 
	• This reduces the likelihood of overfitting, especially if the dataset is relatively small.


### LRCN V1.0:

![image](https://github.com/user-attachments/assets/6f772837-8156-405d-b89d-82d61c7be6df)

#### Inferences:

1. The training loss fluctuates early on and then stabilizes around 0.685. 
	• This suggests that the model is struggling to improve its fit to the training data significantly.
2. The validation loss is lower than the training loss for most epochs. 
	• This can happen when the model's performance on the validation set is better than on the training set, but it could also indicate data leakage or issues with the validation split.
3. Neither training loss nor validation loss improves significantly over epochs, indicating that the model is not effectively learning the patterns in the data.

4. The training accuracy fluctuates slightly and stays constant around 0.55. This indicates that the model struggles to fit the training data.
5. The validation accuracy is static at 0.60 and does not improve. This suggests the model cannot generalize well to unseen data, likely due to poor learning capacity or class imbalance.
6. The lack of improvement in validation accuracy implies that the model may not be learning meaningful temporal patterns.

#### Recommendations:

1. Sequence Length:
	• SEQUENCE_LENGTH = 3. Falls are complex events that might require more frames to fully understand the sequence.
	• Use more frames per video (e.g., SEQUENCE_LENGTH = 10 or SEQUENCE_LENGTH = 20).
2. Overly Simple LRCN Model:
	• The LSTM layer uses only 32 units, which may not be sufficient to capture the temporal dependencies in the data.
	• Use 64 units: model.add(LSTM(64, return_sequences=True))
3. Small Batch Size:
	• A batch size of 4 might lead to noisy gradient updates and hinder the optimization process. 
	• Larger batch sizes often result in more stable learning.
	• Increase the batch size to 8 or 16 to improve the stability of gradient updates.
4. Learning Rate:
	• The Adam optimizer may be using a learning rate that's too high, causing the model to converge sub-optimally.
	• Reduce the learning rate for Adam: optimizer = Adam(learning_rate=0.0001)
5. Dropout Placement:
	• Excessive dropout in convolutional layers (0.25) might lead to underfitting by removing too much information.
	• Use smaller dropout rates (e.g., 0.1 or 0.15) in the convolutional layers to retain more information.
6. Validation Split:
	• Ensure the validation split is stratified to avoid class imbalance in the validation set.


## Version 2.0:

### ConvLSTM V2.0:

![image](https://github.com/user-attachments/assets/1acc6872-08b0-4030-9294-d35b44938ce1)


### LRCN V1.0:
![image](https://github.com/user-attachments/assets/bea3b1f6-af79-4999-aa40-bcee3bcb864c)


