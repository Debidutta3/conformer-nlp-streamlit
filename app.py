import streamlit as st
import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Conformer NLP Demo",
    layout="centered"
)

st.title("🧠 Conformer-Based NLP System")
st.write("Emotion Detection & Intent Classification")

# -----------------------------
# Load label mappings
# -----------------------------
with open("label_mapping.json", "r") as f:
    label_data = json.load(f)

EMOTION_LABELS = label_data["emotion_labels"]
INTENT_LABELS = label_data["intent_labels"]

# -----------------------------
# Model architecture (MUST MATCH TRAINING)
# -----------------------------
class ConformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

        self.attn = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True
        )

        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        x = x + self.ff(x)

        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=~mask
        )
        x = x + attn_out

        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + conv_out

        return self.norm(x)


class TextConformer(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)
        self.encoder = ConformerBlock(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.emotion_head = nn.Linear(dim, len(EMOTION_LABELS))
        self.intent_head  = nn.Linear(dim, len(INTENT_LABELS))

    def forward(self, ids, mask):
        x = self.embed(ids)
        x = self.encoder(x, mask)

        x = self.pool(x.transpose(1, 2)).squeeze(-1)

        return self.emotion_head(x), self.intent_head(x)

# -----------------------------
# Load tokenizer & model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = TextConformer(
        vocab_size=tokenizer.vocab_size,
        dim=128
    )

    model.load_state_dict(
        torch.load("conformer_multitask_nlp.pt", map_location="cpu")
    )
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# -----------------------------
# User input
# -----------------------------
text_input = st.text_area(
    "Enter a sentence:",
    placeholder="I am very happy with the service today!"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        encoded = tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )

        with torch.no_grad():
            emo_logits, intent_logits = model(
                encoded["input_ids"],
                encoded["attention_mask"].bool()
            )

        emo_probs = torch.softmax(emo_logits, dim=1)[0]
        int_probs = torch.softmax(intent_logits, dim=1)[0]

        emo_idx = torch.argmax(emo_probs).item()
        int_idx = torch.argmax(int_probs).item()

        st.subheader("🔍 Prediction Results")

        st.write(
            f"**Emotion:** {EMOTION_LABELS[emo_idx]} "
            f"({emo_probs[emo_idx]*100:.2f}%)"
        )

        st.write(
            f"**Intent:** {INTENT_LABELS[int_idx]} "
            f"({int_probs[int_idx]*100:.2f}%)"
        )

        # -----------------------------
        # Confidence visualizations
        # -----------------------------
        st.subheader("📊 Confidence Distribution")

        st.write("Emotion Confidence")
        st.bar_chart(
            {EMOTION_LABELS[i]: float(emo_probs[i])
             for i in range(len(EMOTION_LABELS))}
        )

        st.write("Intent Confidence")
        st.bar_chart(
            {INTENT_LABELS[i]: float(int_probs[i])
             for i in range(len(INTENT_LABELS))}
        )
