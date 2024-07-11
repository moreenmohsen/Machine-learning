import numpy as np
import cv2
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap

class ExerciseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ExerciseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_dataset(dataset_path: str) -> tuple:
    x = []
    y = []
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return np.array(x), np.array(y)

    for exercise_folder in os.listdir(dataset_path):
        exercise_label = exercise_folder
        exercise_folder_path = os.path.join(dataset_path, exercise_folder)

        if not os.path.isdir(exercise_folder_path):
            print(f"Skipping non-directory {exercise_folder_path}")
            continue

        for video_file in os.listdir(exercise_folder_path):
            video_path = os.path.join(exercise_folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video file {video_path}")
                continue

            frame_count = 0
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose.process(image_rgb)
                if result.pose_landmarks:
                    landmarks = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
                    x.append(landmarks)
                    y.append(exercise_label)
                frame_count += 1
            cap.release()
            print(f"Processed {frame_count} frames from video {video_file}")

    return np.array(x), np.array(y)

def plot_training_loss(minibatch_loss_list, num_epochs, iter_per_epoch, averaging_iterations=200):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(minibatch_loss_list)), minibatch_loss_list, label='Training Loss')
    if len(minibatch_loss_list) >= averaging_iterations:
        smoothed_loss = np.convolve(minibatch_loss_list, np.ones(averaging_iterations)/averaging_iterations, mode='valid')
        plt.plot(range(averaging_iterations-1, len(minibatch_loss_list)), smoothed_loss, label='Smoothed Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(train_acc_list, valid_acc_list):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_acc_list)), train_acc_list, label='Training Accuracy')
    plt.plot(range(len(valid_acc_list)), valid_acc_list, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()

def compute_confusion_matrix(model, x_data, y_data):
    model.eval()
    with torch.no_grad():
        outputs = model(x_data)
        _, preds = torch.max(outputs, 1)
    cm = confusion_matrix(y_data.cpu(), preds.cpu())
    return cm

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    
    # Create the display object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plot the confusion matrix with the Seismic colormap
    cm_display.plot(ax=ax, cmap=plt.cm.seismic, colorbar=False)
    
    # Add text annotations for each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2. else 'black')

    plt.show()



def main():
    dataset_path = 'All data'
    NUM_CLASSES = 4
    x_data, y_data = load_dataset(dataset_path)

    if x_data.size == 0 or y_data.size == 0:
        print("No data loaded. Exiting...")
        return

    print("Shape of x_data (pose landmarks):", x_data.shape)
    print("Shape of y_data (exercise labels):", y_data.shape)

    
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)

    
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    print("Shape of x_train (training data):", x_train.shape)
    print("Shape of y_train (training labels):", y_train.shape)
    print("Shape of x_val (validation data):", x_val.shape)
    print("Shape of y_val (validation labels):", y_val.shape)
    print("Shape of x_test (testing data):", x_test.shape)
    print("Shape of y_test (testing labels):", y_test.shape)

   
    model = ExerciseClassifier(x_train.shape[1], NUM_CLASSES)

  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

   
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()


        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_train_tensor).sum().item()
        accuracy = correct / len(y_train_tensor)

        train_losses.append(loss.item())
        train_accuracies.append(accuracy)

      
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            _, val_predictions = torch.max(val_outputs, 1)
            val_correct = (val_predictions == y_val_tensor).sum().item()
            val_accuracy = val_correct / len(y_val_tensor)
            val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

    
    torch.save(model.state_dict(), 'exercise_classifier.pth')
    
    
    np.save('label_encoder.npy', label_encoder.classes_)

   
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_tensor)
        _, predictions = torch.max(outputs, 1)

  
    correct = (predictions == y_test_tensor).sum().item()
    test_accuracy = correct / len(y_test_tensor)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    
    plot_training_loss(minibatch_loss_list=train_losses,
                       num_epochs=num_epochs,
                       iter_per_epoch=len(x_train_tensor) // 32,  # Assuming batch size of 32
                       averaging_iterations=200)

    
    plot_accuracy(train_acc_list=train_accuracies,
                  valid_acc_list=val_accuracies)


    cm = compute_confusion_matrix(model=model, x_data=x_test_tensor, y_data=y_test_tensor)
    plot_confusion_matrix(cm, class_names=label_encoder.classes_)

if __name__ == '__main__':
    main()
