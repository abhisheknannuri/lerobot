import torchvision.models as models
from torchinfo import summary

# 1. Load the model
model = models.resnet18()

# 2. Print a deeply detailed summary of the architecture
# We specify the input size so it can calculate the data flow and memory usage
summary(
    model, 
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    col_width=20,
    row_settings=["var_names"]
)