from BERT_model import BertModel
import json
    

# load NAFBee.json
file_path = "NAFBee_BERT.json"
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

model = BertModel(requires_grad = True, activation=info_activation)
tokenizer = model.tokenizer

print(model)
print('Network: ', info_network)
print('Activation: ', info_activation)
print('Accuracy: ', info_accuracy)
            
            