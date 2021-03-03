import os
from clean_quant_lib import StaticMnistNet, timer
import torch
import random

input_model_directory = "data"
# print("Input directory:", input_model_directory)

NUM_TESTS = 5000
print("Predictions count for each model:", NUM_TESTS)

SEED = 158123

torch.manual_seed(SEED)
device = torch.device("cpu")
torch.backends.quantized.engine = "qnnpack"

data = torch.load(os.path.join(input_model_directory, "data.pt"))
target = torch.load(os.path.join(input_model_directory, "target.pt"))

DATA_SIZE = data.shape[0]

"""
#################################################################################################
# 01. Load Models:
#################################################################################################
"""
models = {}

# StaticMnistNet Original
staticmnistnet_model = StaticMnistNet()
staticmnistnet_model.load_state_dict(
    torch.load(os.path.join(input_model_directory, "staticmnistnet_state_dict.pt"), map_location=device))
staticmnistnet_model.eval()
models.update({"StaticMnistNet Original": staticmnistnet_model})

# StaticMnistNet Static Quantization with fusing
stquant_fused_staticmnistnet_model = StaticMnistNet()
stquant_fused_staticmnistnet_model.eval()
stquant_fused_staticmnistnet_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
stquant_fused_staticmnistnet_model = torch.quantization.fuse_modules(stquant_fused_staticmnistnet_model,
                                                                     [['conv1', 'relu1'], ['conv2', 'relu2']],
                                                                     inplace=False)
stquant_fused_staticmnistnet_model = torch.quantization.prepare(stquant_fused_staticmnistnet_model, inplace=False)
stquant_fused_staticmnistnet_model(data[0:1])
stquant_fused_staticmnistnet_model = torch.quantization.convert(stquant_fused_staticmnistnet_model, inplace=False)
stquant_fused_staticmnistnet_model.eval()
stquant_fused_staticmnistnet_model.load_state_dict(
    torch.load(os.path.join(input_model_directory, "converted_fused_staticmnistnet_state_dict.pt"),
               map_location=device))
models.update({"StaticMnistNet Static Quantization with fusing": stquant_fused_staticmnistnet_model})

"""
#################################################################################################
# 02. Models speed test:
#################################################################################################
"""

for model_name in models:
    print()
    print(model_name + ":", flush=True)
    with timer():
        with torch.no_grad():
            for iii in range(NUM_TESTS):
                tmp_idx = random.randint(0, DATA_SIZE - 1)
                tmp_data = data[tmp_idx:tmp_idx + 1]
                tmp_target = target[tmp_idx:tmp_idx + 1]
                output = models[model_name](tmp_data)
            # print("Target: %d; Prediction: %d" % (tmp_target.cpu().detach().numpy()[0], output.argmax(dim=1, keepdim=True).cpu().detach().numpy()[0,0]))
