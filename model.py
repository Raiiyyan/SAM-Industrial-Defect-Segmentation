"""
IndustrialSAM — multi-task wrapper around SAM ViT-B that jointly predicts
defect masks and defect class logits in a single forward pass.

Architecture
------------
1. Frozen image encoder + injected IndustrialAdapters (Week 2)
2. Prompt encoder (box prompt)
3. Mask decoder  →  mask_logits (segmentation)
                 →  IoU token (extracted via transformer forward hook)
4. DefectClassifierHead (on IoU token)  →  class_logits (8 classes)

The IoU token (``hs[:, 0, :]``) is the first of the five output tokens
produced by the TwoWayTransformer.  It cross-attends to all image features,
giving it a global representation of the detected object / defect.  SAM
already uses this token to predict mask quality (IoU); we reuse it for
classification, sharing the same computation with zero extra encoder cost.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling.mask_decoder import MaskDecoder

from adapters import IndustrialAdapter
from classifier_head import DefectClassifierHead, CLASS_VOCAB, NUM_CLASSES
from model_setup import (EMBED_DIM, BOTTLENECK_DIM, _ADAPTER_PREFIX,
                         _make_patched_forward)


class IndustrialSAM(nn.Module):
    """Multi-task SAM that outputs both mask logits and class logits.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``sam_vit_b_01ec64.pth``.
    device : str
        Torch device string.
    bottleneck_dim : int
        Adapter bottleneck dimension (default 64).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        bottleneck_dim: int = BOTTLENECK_DIM,
    ):
        super().__init__()
        self.device = device

        # ── Load SAM, freeze image encoder, inject adapters ──────
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.eval()
        for param in sam.image_encoder.parameters():
            param.requires_grad = False

        for block in sam.image_encoder.blocks:
            ad_attn = IndustrialAdapter(EMBED_DIM, bottleneck_dim)
            ad_mlp = IndustrialAdapter(EMBED_DIM, bottleneck_dim)
            block.adapter_attn = ad_attn
            block.adapter_mlp = ad_mlp
            block.forward = _make_patched_forward(block, ad_attn, ad_mlp)

        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder

        # ── Register IoU token hook on the Transformer ────────────
        self._iou_token = None

        def _iou_hook(module, args, output):
            hs, _ = output
            self._iou_token = hs[:, 0, :]

        self._hook_handle = sam.mask_decoder.transformer.register_forward_hook(
            _iou_hook
        )

        # ── Classifier head ────────────────────────────────────────
        self.classifier_head = DefectClassifierHead()

        # ── Class vocabulary ───────────────────────────────────────
        self.class_vocab = CLASS_VOCAB

        # ── Move everything to target device ──────────────────────
        self.to(device)

    def forward(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jointly predict defect mask and class logits.

        Parameters
        ----------
        image : torch.Tensor  (B, 3, 1024, 1024)
            ImageNet-normalised input image (letterboxed).
        boxes : torch.Tensor  (B, 4)  or None
            Bounding-box prompts in pixel coordinates [x1, y1, x2, y2].
            If None, a full-image box [0, 0, 1023, 1023] is used.

        Returns
        -------
        mask_logits : torch.Tensor  (B, 1, 256, 256)
            Raw mask logits (apply sigmoid for binary prediction).
        class_logits : torch.Tensor  (B, 8)
            Raw class logits (feed to CrossEntropyLoss, no softmax).
        """
        B = image.shape[0]

        if boxes is None:
            boxes = torch.tensor(
                [[0, 0, 1023, 1023]], device=self.device
            ).repeat(B, 1)

        # ── Image encoder (batched — handles full batch) ───────────
        image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64)
        self.prompt_encoder.input_image_size = (1024, 1024)
        image_pe = self.prompt_encoder.get_dense_pe()

        # ── Decoder loop (SAM's mask decoder assumes B=1) ─────────
        mask_logits_list, class_logits_list = [], []

        for i in range(B):
            img_emb_i = image_embeddings[i:i + 1]
            box_i = boxes[i:i + 1]

            sparse, dense = self.prompt_encoder(
                points=None, boxes=box_i, masks=None,
            )

            mask_i, _ = self.mask_decoder(
                image_embeddings=img_emb_i,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )

            iou_token = self._iou_token  # captured by transformer hook
            class_i = self.classifier_head(iou_token)

            mask_logits_list.append(mask_i)
            class_logits_list.append(class_i)

        return torch.cat(mask_logits_list, dim=0), torch.cat(class_logits_list, dim=0)

    def get_trainable_params(self) -> list:
        """Return list of parameters that should receive gradients.

        Includes: adapter params (image encoder) + classifier head + any
        other trainable components.
        """
        return [p for p in self.parameters() if p.requires_grad]


# ── Smoke test ──────────────────────────────────────────────────────

def _run_smoke_test():
    """Create IndustrialSAM with dummy data and print output shapes."""
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else (
        "G:/SAM-Industrial-Defect-Segmentation/sam_vit_b_01ec64.pth"
    )

    model = IndustrialSAM(ckpt, device="cpu")
    model.eval()

    B = 2
    dummy_image = torch.randn(B, 3, 1024, 1024)
    dummy_boxes = torch.tensor([
        [100, 100, 500, 500],
        [200, 200, 600, 600],
    ], dtype=torch.float32)

    with torch.no_grad():
        mask_logits, class_logits = model(dummy_image, boxes=dummy_boxes)

    print(f"mask_logits shape:   {mask_logits.shape}   (expected (B, 1, 256, 256))")
    print(f"class_logits shape:  {class_logits.shape}  (expected (B, 8))")
    print(f"class_vocab:         {model.class_vocab}")
    assert mask_logits.shape == (B, 1, 256, 256), f"mask shape mismatch: {mask_logits.shape}"
    assert class_logits.shape == (B, 8), f"class shape mismatch: {class_logits.shape}"
    print("\nSmoke test passed (all shapes correct).")

    # Print adapter counts for verification
    adapter_params = sum(
        p.numel() for n, p in model.named_parameters() if _ADAPTER_PREFIX in n
    )
    head_params = sum(p.numel() for p in model.classifier_head.parameters())
    total_trainable = sum(p.numel() for p in model.get_trainable_params())
    print(f"\nAdapter params:      {adapter_params:>8,}")
    print(f"Classifier head:     {head_params:>8,}")
    print(f"Total trainable:     {total_trainable:>8,}")


if __name__ == "__main__":
    _run_smoke_test()
