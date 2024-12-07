import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

def load_clip_model():
    model_name = "ViT-B-32"
    pretrained = "openai"
    model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = get_tokenizer(model_name)
    return model, preprocess_val, tokenizer

def encode_text(model, tokenizer, text_query):
    with torch.no_grad():
        tokens = tokenizer([text_query])
        text_emb = model.encode_text(tokens)
        text_emb = F.normalize(text_emb, p=2, dim=1).cpu().numpy()[0]
    return text_emb

def encode_image(model, preprocess, image):
    # image should be a PIL Image
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0)
        img_emb = model.encode_image(img_tensor)
        img_emb = F.normalize(img_emb, p=2, dim=1).cpu().numpy()[0]
    return img_emb
