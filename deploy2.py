# File: streamlit

import streamlit as st
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

IMG_SIZE = 224
NUM_CLASSES = 5
MODEL_PATH = "diabetic_retinopathy_model.pth"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

class DRModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DRModel, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

@st.cache(allow_output_mutation=True)
def load_model():
    model = DRModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)  

def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
    return predicted.item(), probabilities

def main():
    st.title("Diabetic Retinopathy Classification")
    st.markdown("This application classifies retinal images into one of the five categories of diabetic retinopathy severity.")

    uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        MAX_WIDTH = 300
        aspect_ratio = image.height / image.width
        new_height = int(MAX_WIDTH * aspect_ratio)
        resized_image = image.resize((MAX_WIDTH, new_height))
        st.image(resized_image, caption="Uploaded Image", use_column_width=False, width=0.5)
        
        st.write("Analyzing the image...")
        preprocessed_image = preprocess_image(image)
        predicted_class, probabilities = predict(preprocessed_image, model)
        
        st.write(f"### Predicted Severity: {CLASS_NAMES[predicted_class]}")
        st.write("### Probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {probabilities[i].item() * 100:.2f}%")
        
        descriptions = [
            "No signs of diabetic retinopathy.",
            "Mild non-proliferative diabetic retinopathy.",
            "Moderate non-proliferative diabetic retinopathy.",
            "Severe non-proliferative diabetic retinopathy.",
            "Proliferative diabetic retinopathy."
        ]
        st.markdown(f"#### Description: {descriptions[predicted_class]}")

if __name__ == "__main__":
    main()
