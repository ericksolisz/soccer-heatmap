import cv2
import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

class CLIPEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        self.model.eval()

    def encode(self, crop_bgr):
        # BGR → RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Preprocesado del modelo
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)
            emb = out.last_hidden_state[:, 0, :]   # CLS token

        # Devolver vector 1D
        return emb.cpu().numpy().squeeze()
