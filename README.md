# Edge-Based-Acoustic-Event-Detection-for-Water-Activities


# Problem statement

To detect and classify water based activities based on audio. 

## Motivation and relavance.

The ultimate vision and goal was to estimate the water usage for each activity based on audio. Generally whenever one wants to estimate the water usage we use a a 












#  Data Collection

This was the most crucial step since we had no available datasets to work with. We created our own dataset and we had to go through multiple iterations to do it.

## Methodology of data collection

Primarily we first collected two sets of data that was finally used in the development and deployment of the model.

- **First (Fixed Evironment)** We collected data of three activities `HandWashing` `Free Running Tap` `Idle (No activity)`. All the three activities were done in a single washroom, in a single wash basin.
- The audio was recorded using a mobile phone at 48kHz. The mobile phone was placed at a single location.
- For each activity we recorded a continuous strech of about 10 - 15 mins of data.
- From the above collected data we also generated a new data of 16Hz using audacity incase we want to specifically train for the aurdirino devices.

- **Second (Variable Environment)** We collected data of 5 avtivities `Handwashing` `Free Running Tap` `Idle (No activity)` `Filling the bottle` `Washing Utensil`. Here we have increased the number of activities.
- The audio was recorded using a mobile phone at 44.1 kHz. We introduced variability of environment, distace of the recording device (phone).
- We collected data across 10 different washrooms. We varied the distance of the recording device by few 10 cm to around 50 cm. We also varied the orientation of the recording device
- For each activity we collected data for about 2-3 mins. Only exception being Filling the bottle, since each bottle filled up within less the 15 seconds.

## Challenges and other failed attempts

During the course of data collection we encountered several challenges, some of them which we discuss below.

- Firstly our initial attempt was to record multiple activities in a single strech, and the annotate the time stamps of the activities to avoid renaming and tranferring large number of files. But the first iteration proved that annotation took too much time. We then moved on to record a one activity per file.

- Secondly while collecting data on a single large file (as we did for data in a single washroom) we realized that doing a few activities like washing hands (leading to pruney hands) and washing utensils caused discomfort. So we moved on to collect short segments of data (2-3mins)

- Since the data was collected in washrooms, privacy was a concern. We ensured that no one was using the washroom when the data was being collected mostly collecting the data at night times.

- While collecting data we were faced with the dillema of water wastage. So we tried to minimize the water wastage (by bare minimum) by using water filled in bottles to water the plants nearby.
# Model Development

## Iteration 1: The Background Noise Problem
**The Challenge:** Initially, we recorded data as continuous 2 to 3-minute audio files for each activity. When we tested the first model, it failed on new data. The model was memorizing the background room noise rather than the actual water sounds.

**The Change:** We split the audio into smaller chunks and balanced the training data. We capped every class to the exact same number of audio chunks to prevent the model from guessing based on class frequency.

**The Result:** The model stopped relying on background noise and began learning the acoustic patterns of the target sounds.

---

## Iteration 2: The Importance of Feature Engineering
**The Challenge:** The mobile phone records audio at 44.1 kHz, meaning a 3-second clip contains over 132,000 raw data points. Feeding this raw data directly into a model requires significant memory and makes pattern recognition difficult.

**The Change:** We shifted our focus to feature engineering. Instead of raw audio waves, we used Fast Fourier Transforms to convert the sound into frequency heatmaps (Mel Spectrograms) and compressed acoustic features (MFCCs).

**The Result:** The data became cleaner and more compressed. The model received a structured map of the audio frequencies rather than a dense raw wave.

---

## Iteration 3: Testing the Fully Connected Neural Network (FCNN)
**The Challenge:** Using our engineered features, we initially built a Fully Connected Neural Network (FCNN). However, FCNNs look at fixed temporal positions. If a splash occurred at second 1 during training, the network struggled to recognize a splash at second 2 during testing. Additionally, connecting all neurons resulted in a large number of parameters.

**The Change:** We realized a time-independent architecture was necessary for this audio classification task to handle variable event timings.

**The Result:** We moved away from the FCNN to find a model capable of scanning audio sequentially.

---

## Iteration 4: Switching to a 2D CNN
**The Challenge:** We needed to resolve the rigid timing issue of the FCNN while keeping the model size small enough for a phone.

**The Change:** We adopted a 1D Convolutional Neural Network (1D CNN). Instead of analyzing the entire audio clip at once, a 1D CNN uses a sliding window to scan the features over time.

**The Result:** The 1D CNN was able to detect sounds regardless of when they occurred in the clip. This change also reduced the overall model size, making it practical for mobile Edge AI.

---

## Iteration 5: Edge Impulse

### 1. Impulse Configuration

The impulse defines how raw audio is segmented into samples.

- **Window size**: 2000 ms  
- **Window stride**: 1000 ms  
- **Input type**: Time-series audio  

This step converts continuous audio into fixed-length examples suitable for feature extraction.

---

### 2. DSP Block: MFCC (Feature Extraction)

Mel Frequency Cepstral Coefficients (MFCCs) are used to represent audio features.

#### MFCC Parameters
- Number of MFCC coefficients: 13  
- Number of mel filter banks: 32  
- Frame length: 25 ms  
- Frame stride: 20 ms  
- FFT length: 2048  
- Low frequency bound: 80 Hz  
- High frequency bound: Nyquist  
- Pre-emphasis coefficient: 0.98  

Each audio window is transformed into a 2D MFCC feature map for learning.

---

### 3. Learning Block: Audio Classifier

A **2D Convolutional Neural Network (CNN)** is used for classification.

#### Model Architecture
- Reshape MFCC features into a 2D format
- Convolutional layers:
  - Conv2D (8 filters, 3×3 kernel, ReLU)
  - MaxPooling + Dropout
  - Conv2D (16 filters, 3×3 kernel, ReLU)
  - MaxPooling + Dropout
- Fully connected output layer with **softmax activation** (5 classes)

<p align="center">
  <img src="images/2d_cnn_architecture_1.png" alt="2D CNN Architecture" width="300">
</p


#### Training Details
- Loss function: Categorical Cross-Entropy  
- Optimizer: Adam  
- Training includes internal shuffling and regularization  
- Optional audio data augmentation (SpecAugment)

---

### 4. Model Evaluation

Edge Impulse automatically evaluates the model using:
- Test accuracy
- Confusion matrix
- Per-class performance metrics

This allows validation of model performance across all activity classes.

#### Testing metrics:
<p align="center">
  <img src="images/testing_metrics.png" alt="metrics" width="500">
</p

#### Confusion Matrix:
<p align="center">
  <img src="images/testing_confusion_matrix.png" alt="metrics" width="500">
</p

---

### **Deployment on Mobile Phone**

The trained model was deployed on a mobile application to enable real-time detection of water-based activities using acoustic signals. The application records audio using the smartphone microphone, processes the input, and predicts activities such as handwashing, bottle filling, idle and utensils cleaning. The results are displayed within the app, providing real-time insights into user activity.

 During live testing, the Flutter mobile app failed to produce accurate predictions. The model was trained to expect engineered features (Spectrograms), but the app was sending raw 44.1 kHz audio, as replicating the feature engineering math natively in Flutter proved difficult.

**The Change:** We embedded the feature engineering directly inside the TensorFlow model by adding a custom preprocessing layer at the front of the network.

**The Result:** The mobile app now records 13 second chunks of raw audio and passes it directly to the model. The model handles its own feature extraction internally, which resolved the mismatch and stabilized the live predictions.



The system relied on a server-based approach, where recorded audio was sent to a backend for processing and prediction. However, this introduced latency and dependency on internet connectivity. To address this limitation, an alternative version of the application was developed to perform on-device inference, allowing direct processing on the mobile device without requiring server communication but it did not completed in mean time.

Users can download and run the application by accessing the **`water_app`** folder from the project GitHub repository. The app can be installed on a mobile device by enabling developer mode and USB debugging, connecting the phone to a laptop, and running the application using Flutter. Detailed setup and usage instructions are provided in the **`README.md`** file included in the repository.

The trained 2D CNN model can be deployed directly from Edge Impulse to:
- Android Phone: Using the edge impulse android library.
- Edge Devices: Using the edge impulse arduino based library and eon compiler.
- On phones we can directly deploy the model by scanning this QR code.

<p align="center">
  <img src="images/qr.png" alt="qr" width="200">
</p


#### Comparison of Quantized and Unquantized Models Deployment
<p align="center">
  <img src="images/deployment.png" alt="deployment" width="500">
</p


---

---

### **Challenges**

* Limited experience in mobile application development.
* Dependency on server-based inference leading to latency.
* Difficulty in implementing on-device model inference.
* Runtime errors and instability in the mobile application.
* Challenges in achieving consistent real-time predictions.


## Deployment on Aurdirino device.










<!-- ![alt text](images/2d_cnn_architecture.png) -->










<img src="drawing.jpg" alt="drawing" width="200"/>