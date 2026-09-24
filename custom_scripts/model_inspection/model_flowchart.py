import torch
import torchvision.models as models
from torchviz import make_dot

# 1. Load the pre-defined ResNet-18 model from PyTorch
model = models.resnet18()

# 2. Create a dummy input tensor (Batch Size: 1, Channels: 3, Height: 224, Width: 224)
# This is the standard input size for ResNet models
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Pass the dummy input through the model to generate the computational graph
output = model(dummy_input)

# 4. Generate the architecture diagram
# We pass the output tensor and the model parameters to label the nodes
architecture_graph = make_dot(output, params=dict(model.named_parameters()))

# 5. Save the diagram to a file (it will save as 'resnet18_full_architecture.pdf')
architecture_graph.format = 'pdf' # You can change this to 'png'
architecture_graph.render("resnet18_full_architecture")

print("Visual diagram saved as resnet18_full_architecture.pdf!")