# Vehicle Behavior Detection and Classification at Intersections

This repository contains the full code for my bachelor thesis, which focuses on detecting vehicles at intersections and classifying their maneuvers (left turn, right turn, straight) using deep learning and trajectory analysis.

## Thesis Paper
 
The detailed approach and evaluation can be found in [Will include the link to the paper later].

## Project Overview

- **Detection & Tracking:** Uses YOLOv8 to detect vehicles and log trajectory data to CSV files.
- **Preprocessing:** Crops and normalizes trajectories, computes per-point angles, and prepares data splits.
- **Model Training:** Trains an LSTM-based model to classify vehicle trajectories.
- **Live Classification:** Runs detection and classification in real time on video footage.

## Setup

### 1. Clone the repository

Clone the repo:
```bash
git clone https://github.com/tolyy-vu/vehicle-behaviour-classification-on-intersections.git
```

### 3. Create and activate a conda environment
```bash
conda create -n vehicle-detection python=3.9
conda activate vehicle-detection
```

### 4. Install requirements
```bash
pip install -r requirements.txt
```
Note: If you plan to use GPU acceleration, ensure that your PyTorch installation supports CUDA.

## Running the pipeline

### 1. Detection & Trajectory Logging
Run the detection script to process video footage and generate a CSV file with vehicle trajectories and direction labels.
You can see the example datasets on the folder dataset/
```bash
python main.py
```
Update the video_path variable inside the script to point to your desired video file. The output CSV file will be saved in the dataset/ directory.
```python
show_debug
```
Set the above flag to True in main.py if you want to see the debug window (it is set to true by default).

### 2. Preprocessing
Prepare trajectory data for training or classification. This step crops each trajectory to 120 points, computes angles, normalizes features, and splits the data into train/validation/test sets.

```bash
python preprocessing.py
```
Ensure that CSV_PATH in the script points to your generated CSV file. The output .npz file will be saved in dataset/full_dataset/.

### 3. Training
Train the LSTM model using the preprocessed data.
```bash
python model.py
```
This script trains the bidirectional LSTM on the preprocessed data using balanced class weights.

Saves plots showing training loss and validation accuracy (accuracy_loss_over_epochs.png) and a test set confusion matrix (confusion_matrix_test.png).

Saves the final trained model as model.pth.

### 4. Live Classification
After detection and preprocessing, run real-time video classification:
```bash
python live_classification.py
```
Update the video path in the script to point to your video file or stream. The system displays live predictions on screen, showing object IDs and predicted maneuver labels.

## Important Notes:
1. Best results are achieved with top-down intersection views similar to this dataset.
<img width="400" height="272" alt="example" src="https://github.com/user-attachments/assets/630e885b-59bd-4d61-b543-395cac95b61b" />
#### An example of usable footage.


2. Ensure model.pth exists before running live classification.

3. main.py will generate multiple seperate CSV files, it is advised to gather all of them in a new CSV file manually before preprocessing.

## Contact
For questions or further information, and if you use this code or build upon it, please contact: m.a.dolgun@student.vu.nl or dolgunmertali@gmail.com for questions and citations.





