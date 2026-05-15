import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

CLASSES = ['Banded_Chlorosis', 'Brown_Spot', 'Dried_Leaves', 'Eye_Spot', 'Grassy_Shoot', 'Healthy', 'Leaf_Scorch', 'Mawa', 'Mites', 'Mosaic', 'Nitrogen_Deficit', 'Pokkah_Boeng', 'Potassium_Deficit', 'Red_Leaf_Spot', 'Red_Rot', 'Red_Spot', 'Ring_Spot', 'Rust', 'Sett_Rot', 'Smut', 'Yellow_Disease']

class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DinoClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Linear(384, num_classes)
        
    def forward(self, x):
        outputs = self.backbone(x)
        # Usamos pooler_output para coincidir con el script de entrenamiento
        return self.classifier(outputs.pooler_output)

print("[*] Test de Visión DINOv2 (MODO CPU - Diagnóstico de Precisión)...")
device = torch.device("cpu")
model = DinoClassifier(len(CLASSES)).to(device)
model.load_state_dict(torch.load("dinov2_classifier_best.pth", map_location=device), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

test_img = "ultimate_dataset/val/Healthy/bd_1695029451471.jpg"
img_pil = Image.open(test_img).convert('RGB')
img_t = transform(img_pil).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_t)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    
    # Obtener los top 5 resultados
    top5_probs, top5_indices = torch.topk(probs, 5)

print("\n" + "="*50)
print("🟢 DINOv2 ANALYZER - DETALLES TOP 5")
print("="*50)
print(f"IMAGEN: {test_img}")
for i in range(5):
    class_name = CLASSES[top5_indices[0][i].item()]
    confidence = top5_probs[0][i].item() * 100
    print(f"{i+1}. {class_name}: {confidence:.2f}%")
print("="*50)
