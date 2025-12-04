import torch
from torchvision import transforms, models
import torch.nn as nn
dummy_input = torch.randn(1, 3, 224, 224)

class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained MobileNetV3-Large
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # Replace classifier with a quaternion head
        in_features = backbone.classifier[0].in_features

        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(0.1),

            nn.Linear(256, 4)   # quaternion output
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