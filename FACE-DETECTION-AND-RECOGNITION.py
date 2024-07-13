import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn
from sklearn.preprocessing import normalize

class FaceDetector:
    def __init__(self, model_path):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def recognize_face(self, img, face):
        x, y, w, h = face
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (112, 112))
        face_tensor = self.transform(face_img).unsqueeze(0)
        with torch.no_grad():
            feature = self.model(face_tensor)
        return feature.numpy()

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 512)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = nn.Sequential(*list(model.children())[:-1])
    return model

def compare_faces(feature1, feature2):
    feature1 = normalize(feature1)
    feature2 = normalize(feature2)
    similarity = np.dot(feature1, feature2.T)
    return similarity

if __name__ == "__main__":
    detector = FaceDetector('arcface_model.pth')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            feature = detector.recognize_face(frame, face)
            similarity = compare_faces(feature, known_feature)
            if similarity > 0.5:
                cv2.putText(frame, 'Known', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
