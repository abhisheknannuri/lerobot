import torch
import torchvision.models as models

model = models.resnet18()
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy_input, "resnet18.onnx")