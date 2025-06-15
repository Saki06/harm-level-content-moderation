# model.py
import torch
import torch.nn as nn

class FlexibleCLIPClassifier(nn.Module):
    def __init__(self, clip_model, mode="both"):
        super().__init__()
        self.clip = clip_model
        self.mode = mode
        self.classifier = nn.Sequential(
            nn.Linear(self.clip.config.projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3 harm levels: Low, Medium, High
        )

    def forward(self, **kwargs):
        if self.mode == "both":
            img_feat = self.clip.get_image_features(pixel_values=kwargs["pixel_values"])
            txt_feat = self.clip.get_text_features(input_ids=kwargs["input_ids"],
                                                   attention_mask=kwargs["attention_mask"])
            embedding = (img_feat + txt_feat) / 2
        elif self.mode == "image":
            embedding = self.clip.get_image_features(pixel_values=kwargs["pixel_values"])
        elif self.mode == "text":
            embedding = self.clip.get_text_features(input_ids=kwargs["input_ids"],
                                                    attention_mask=kwargs["attention_mask"])
        else:
            raise ValueError("Invalid mode")
        return self.classifier(embedding)
