import torch
import hydra
import pandas as pd
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from omegaconf import DictConfig
from models.model import SwinBart
import os
import unicodedata

def draw_caption(image: Image.Image, pred: str, gt: str) -> Image.Image:
    draw = ImageDraw.Draw(image)

    # Fallback to default font if arial is not found
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    def safe_text(text):
        # Normalize text and remove unsupported characters
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    def overlay(pos, text, col, bgcolor):
        safe = safe_text(text)
        size = draw.textsize(safe, font=font)
        draw.rectangle([*pos, pos[0]+size[0]+4, pos[1]+size[1]+4], fill=bgcolor)
        draw.text((pos[0]+2, pos[1]+2), safe, fill=col, font=font)

    overlay((10, 10), f"Pred: {pred}", "white", "black")
    overlay((10, 40), f"GT:   {gt}", "white", "darkgreen")
    return image


@hydra.main(config_path="configs", config_name="default")
def inference(cfg: DictConfig):
    # Set device
    device = torch.device(cfg.trainer.device)

    # Initialize and load model
    model = SwinBart(cfg).to(device)
    model.load_state_dict(torch.load("/home/hov1syv/Desktop/Personal/research/video-commentary-ai/checkpoints/best_model.pth", map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground-truth captions
    caption_df = pd.read_csv(cfg.inference.captions_csv)
    all_captions = caption_df["caption"].values
    print(len(all_captions))
    frames = [f"frame_{i:04d}.jpg" for i in range(len(all_captions))]
    gt_captions = dict(zip(frames, all_captions))

    # Randomly select frames
    image_dir = Path(cfg.inference.image_dir)
    all_frames = list(image_dir.glob("frame_*.jpg"))
    sampled_frames = random.sample(all_frames, k=cfg.inference.num_samples)

    for img_path in sampled_frames:
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                images=image_tensor,
                max_length=cfg.inference.max_length,
                num_beams=cfg.inference.num_beams
            )
        pred_caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        gt_caption = gt_captions.get(img_path.name.split("/")[-1], "[No ground truth]")

        print(f"{img_path.name}\nPred: {pred_caption}\nGT:   {gt_caption}\n")

        # Draw and save result
        output_img = draw_caption(image.copy(), pred_caption, gt_caption)
        output_img.save(output_dir / img_path.name)



if __name__ == "__main__":
    inference()
