import torch
from torchvision import transforms, models
import torch.nn as nn
dummy_input = torch.randn(1, 3, 224, 224)

class HardSwishManual(nn.Module):
    def forward(self, x):
        # hswish(x) = x * relu6(x + 3) / 6
        return x * nn.functional.relu6(x + 3) / 6


class HardSigmoidManual(nn.Module):
    def forward(self, x):
        # hsigmoid(x) = relu6(x + 3) / 6
        return nn.functional.relu6(x + 3) / 6


# -------------------------------------------------------
# Module replacer
# -------------------------------------------------------

def replace_hard_ops(module):
    """
    Recursively replace Hardswish → HardSwishManual
    and Hardsigmoid → HardSigmoidManual
    """
    for name, child in module.named_children():

        # Replace Hardswish
        if isinstance(child, nn.Hardswish):
            module.add_module(name, HardSwishManual())

        # Replace Hardsigmoid
        elif isinstance(child, nn.Hardsigmoid):
            module.add_module(name, HardSigmoidManual())

        # Recurse
        else:
            replace_hard_ops(child)


# -------------------------------------------------------
# Pose Model
# -------------------------------------------------------

class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

        # Replace all hard ops in backbone
        replace_hard_ops(backbone)

        # Replace classifier head
        in_features = backbone.classifier[0].in_features

        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            HardSwishManual(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            HardSwishManual(),
            nn.Dropout(0.1),

            nn.Linear(256, 4),
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

model = PoseModel()

# Load your trained weights
checkpoint_path = "pose_model_final.pt"  # replace with your .pt file
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