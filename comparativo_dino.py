import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import AutoModel
import os

CLASSES = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']

class DinoCLS(nn.Module):
    def __init__(self, num_classes):
        super(DinoCLS, self).__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Linear(384, num_classes)
    def forward(self, x, mode="pooler"):
        outputs = self.backbone(x)
        if mode == "pooler":
            return self.classifier(outputs.pooler_output)
        else:
            return self.classifier(outputs.last_hidden_state[:, 0, :])

device = torch.device("cpu")
model = DinoCLS(len(CLASSES)).to(device)
model.load_state_dict(torch.load("dinov2_classifier_best.pth", map_location=device), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder("ultimate_dataset/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

print("[*] Comparando MODOS de extracción...")
correct_pooler = 0
correct_cls = 0
total = 50

with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_loader):
        if i >= total: break
        inputs = inputs.to(device)
        
        # Modo Pooler
        out_p = model(inputs, mode="pooler")
        _, pred_p = torch.max(out_p, 1)
        if pred_p.item() == labels.item(): correct_pooler += 1
        
        # Modo CLS Puro
        out_c = model(inputs, mode="cls")
        _, pred_c = torch.max(out_c, 1)
        if pred_c.item() == labels.item(): correct_cls += 1

print("\n" + "="*50)
print(f"🎯 Precisión Pooler: {(correct_pooler/total)*100:.2f}%")
print(f"🎯 Precisión CLS Puro: {(correct_cls/total)*100:.2f}%")
print("="*50)
