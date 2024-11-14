import gradio as gr
import torch
from torchvision import transforms, models, utils

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load your model (modify if necessary based on your model architecture)
def load_model(model_checkpoint_path):
    # Load the model's state_dict (weights)
    model = models.resnet18(pretrained=True)
    # Modify the final fully connected layer
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# Function to predict the image and get confidence score
def predict_image(image, model):
    image = test_transform(image).unsqueeze(0)
    image = image.to(device)
    model = model.to(device)

    # Get model's prediction and confidence scores
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)  # Get class probabilities
        confidence, predicted_class = torch.max(probs, 1)  # Get the highest probability and corresponding class
    
    confidence_score = confidence.item() * 100  # Convert to percentage
    predicted_class = predicted_class.item()
    
    return predicted_class, confidence_score

# Gradio Interface function
def inference(image, model):
    predicted_class, confidence_score = predict_image(image, model)
    return f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}%"

# Create Gradio Interface
def create_interface():
    # Interface for uploading the model and predicting
    def upload_and_predict(image):
        model = load_model("checkpoint.pth") 
        print(inference(image, model))
        return inference(image, model)

    interface = gr.Interface(
        fn=upload_and_predict,
        inputs=gr.Image(type="pil", label="Upload Image for Prediction"),
        outputs="text",
        live=True
    )

    return interface



# Launch Gradio app
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()