import torch
import torch.nn as nn



class LoRAModule(nn.Module):
    def __init__(self, original_layer, r):
        super(LoRAModule, self).__init__()
        self.original_layer = original_layer
        self.r = r
        
        self.lora_A = nn.Linear(original_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_layer.out_features, bias=False)
        
    def forward(self, x):
        return self.original_layer(x) + self.lora_B(self.lora_A(x))

class LoRAWrapper(nn.Module):
    def __init__(self, original_model, r=4):
        super(LoRAWrapper, self).__init__()
        self.original_model = original_model
        self.r = r
        
        # Replace layers containing "proj" with LoRA-enhanced layers
        self.modify_layers(self.original_model)

    def modify_layers(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and ('proj' in name or "out_layer" in name):
                setattr(module, name, LoRAModule(child, self.r))
            else:
                self.modify_layers(child)

    def forward(self, x):
        return self.original_model(x)