import numpy as np
import cv2
import mediapipe as mp
import torch
from ai import ExerciseClassifier  

def load_model(model_path: str, input_size: int, num_classes: int):
    model = ExerciseClassifier(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_label_encoder(label_encoder_path: str):
    classes = np.load(label_encoder_path, allow_pickle=True)
    return classes

def process_frame(frame):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    if result.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
        return landmarks
    else:
        return None

def classify_frame(model, frame, label_encoder):
    landmarks = process_frame(frame)
    if landmarks is None:
        return "No valid pose detected", False
    
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        _, prediction = torch.max(outputs, 1)
        class_name = label_encoder[prediction.item()]
        return class_name, True

def main():
    
    input_size = 66  # 33 landmarks * 2 (x, y)
    num_classes = 4  # Adjust based on your dataset

    model_path = 'exercise_classifier.pth'
    label_encoder_path = 'label_encoder.npy'

  
    model = load_model(model_path, input_size, num_classes)
    label_encoder = load_label_encoder(label_encoder_path)

   
    cap = cv2.VideoCapture(0)

    frame_limit = 500   
    frame_count = 0
    correct_predictions = 0

    while frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

      
        classification_result, is_valid = classify_frame(model, frame, label_encoder)
        print(f'Classification Result: {classification_result}')

        if is_valid:
            correct_predictions += 1
        
        accuracy = correct_predictions / (frame_count + 1)

       
        cv2.putText(frame, f'Class: {classification_result}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Exercise Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
