#  Visual-Textual Harm-Level Based Autonomous Content Moderation Agent

This project is an AI-powered content moderation system that detects and scores harmful content in social media posts — analyzing **text**, **images**, or a combination of both. Built using the [Facebook Hateful Memes Dataset](https://www.kaggle.com/datasets/facebook/hateful-memes), the model classifies content as **Low**, **Medium**, or **High** harm, and recommends enforcement actions accordingly.

---

## 📌 Features

-  Multi-modal content moderation using **CLIP (text + image)**
-  Harm level classification: `Low`, `Medium`, `High`
-  Autonomous moderation logic:
  - `High` → alert + auto-remove
  - `Medium` → auto-blur/sanitize
  - `Low` → allow/flag
-  Supports individual text, image, or image+text evaluation
-  Live testing via CLI or Streamlit UI
-  Handles class imbalance with weighted loss training
  

---

