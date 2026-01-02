import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# 1. Re-initialize the model structure
# (Ensure the Net class from your provided code is defined in this scope)
device = torch.device("cpu")
model = Net().to(device)

# 2. Load the trained weights
# Make sure you have run the training script with --save_model first
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

def predict_image(image_path):
    # 3. Pre-processing
    # MNIST images are 28x28 grayscale. We must match the training transforms.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load image and add batch dimension (1, 1, 28, 28)
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        output = model(img_tensor)

        # [Optional] Convert Log-Softmax to standard Probabilities
        probs = torch.exp(output)
        
        print("Model output logits:", probs)
        
        # The model uses log_softmax, so we take the index of the highest value
        prediction = probs.argmax(dim=1, keepdim=True).item()
        
    return prediction

# Usage
image_file = "/home/jackyyeh/playground/Tiny-Neural-Inference-Engine/inputs/testSet/img_6.jpg" 
result = predict_image(image_file)
print(f"Predicted Digit: {result}")