import json
from models import *
import torch

# load NAFBee.json
file_path = "NAFBee_VGG19.json"
with open(file_path, "r") as json_file:
    nafbee = json.load(json_file)
    

# Obtain the No. 1 network information
# NAFBee provides 11 architectures (No.1-No.11)
info = nafbee["1"]
print(info)

# all information of No.1 network
info_network = info["network"]
info_activation = info["activation"]
info_accuracy = info["accuracy"]

# Define the model on pytorch
if "VGG" in info_network:
    model = VGG(info_network, info_activation)
print(model)

print('No.1 Base network: {}, Activation: {}, Accuracy:{}%'.format(info_network,info_activation,info_accuracy))







