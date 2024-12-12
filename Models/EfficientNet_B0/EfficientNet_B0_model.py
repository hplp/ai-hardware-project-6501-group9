import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_b0

# Load the EfficientNet-B0 model
model = efficientnet_b0()
# Modify the first convolutional layer for grayscale input
model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# Modify the classifier for binary classification
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
# Load the trained weights
model.load_state_dict(torch.load("drowsiness_efficientnet_b0.pth"))  # Replace with the correct checkpoint file
model.eval()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalization for grayscale images
])

# Helper function to preprocess an image and predict
def predict_drowsiness(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)

    # Transform and prepare input
    transformed_image = transform(pil_image).unsqueeze(0).to(device)

    # Predict using the model
    with torch.no_grad():
        output = model(transformed_image)
        _, prediction = torch.max(output, 1)
    return prediction.item()  # 1 = Open eyes, 0 = Closed eyes

# Real-time webcam detection
def detect_drowsiness_from_webcam():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face)

            for (ex, ey, ew, eh) in eyes:
                eye = face[ey:ey + eh, ex:ex + ew]
                label = predict_drowsiness(eye)
                text = "Open" if label == 1 else "Closed"

                # Draw a rectangle and label around the eye
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), color, 2)
                cv2.putText(face, text, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the result
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to run real-time detection or image detection
if __name__ == "__main__":
    mode = input("Choose mode: 'webcam' for real-time detection,").strip().lower()

    if mode == "webcam":
        detect_drowsiness_from_webcam()
    else:
        print("Invalid mode! Choose 'webcam'")
