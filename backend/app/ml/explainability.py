"""Grad-CAM (vision) and gradient-attribution (text) explanation helpers.

Both methods are technically valid post-hoc explanations:
  * Text: gradient x input token attribution on the fine-tuned DistilBERT.
  * Vision: Grad-CAM (Selvaraju et al., 2017) on the CNN's last conv block.
No explanation is ever fabricated: if a model does not expose the required
layers/inputs, the helper raises and callers report explanations as
unavailable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def text_token_attribution(model, tokenizer, text: str, max_length: int = 64,
                           top_k: int = 5) -> Dict:
    """Gradient x input attribution for the predicted class.

    Returns tokens with the largest attribution magnitudes, signed.
    """
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    embeds = model.get_input_embeddings()(input_ids)

    embeds = embeds.detach().requires_grad_(True)
    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    logits = outputs.logits
    pred = int(logits.argmax(dim=-1).item())
    score = logits[0, pred]
    model.zero_grad(set_to_none=True)
    score.backward()

    grad = embeds.grad.detach()
    saliency = (grad * embeds.detach()).sum(dim=-1)[0]  # (seq_len,)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    mask = attention_mask[0].tolist()

    scored: List[Tuple[str, float]] = [
        (tok, float(s)) for tok, s, m in zip(tokens, saliency.tolist(), mask) if m == 1
    ]
    top = sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    # Strip special tokens from the explanation
    top = [(t, s) for t, s in top if t not in ("[CLS]", "[SEP]", "[PAD]")]
    return {
        "predicted_class_index": pred,
        "attributions": [{"token": t, "score": round(s, 5)} for t, s in top],
        "method": "gradient_x_input",
    }


def grad_cam(model, image_batch, target_class: int, conv_layer_name: str = "conv_features") -> np.ndarray:
    """Grad-CAM heatmap (48x48, float 0..1) for a 4D grayscale batch of size 1."""
    import torch

    layer = getattr(model, conv_layer_name, None)
    if layer is None:
        raise ValueError(f"Model has no layer named {conv_layer_name!r} for Grad-CAM")

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_module, _inp, out):
        activations.append(out.detach())

    def bwd_hook(_module, _gin, gout):
        gradients.append(gout.detach())

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        x = image_batch.clone().requires_grad_(True)
        logits = model(x)
        score = logits[0, target_class]
        score.backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not fire")
        acts = activations[0][0]          # (C, H, W)
        grads = gradients[0][0]           # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # global average pooling
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        return cam_np.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()
