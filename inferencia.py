import torch
import torch.nn as nn

# Define a simple CNN class (assuming this matches the saved model)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14x14 -> 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load('cnn_model.pth'))
model.eval()

# Short test
with torch.no_grad():
    dummy_input = torch.randn(1, 1, 28, 28)  # Assuming MNIST
    output = model(dummy_input)
    print("Test output:", output)
