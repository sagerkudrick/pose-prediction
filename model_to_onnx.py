import torch
from torchvision import transforms, models
import torch.nn as nn
dummy_input = torch.randn(1, 3, 224, 224)

class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        
        # Replace adaptive avg pool with fixed 7x7 avg pool (ONNX compatible)
        backbone.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # Replace fully connected layers with ONNX-safe architecture
        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

model = PoseModel()

# Load your trained weights
checkpoint_path = "pose_model_best.pt"  # replace with your .pt file
state_dict = torch.load(checkpoint_path, map_location='cpu')  # use map_location='cpu' if no GPU
model.load_state_dict(state_dict)

model.eval()

torch.onnx.export(model, dummy_input, "model.onnx",
                  opset_version=8,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  external_data=False,
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)