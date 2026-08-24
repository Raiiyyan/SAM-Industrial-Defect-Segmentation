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

Training runs at 512x512 for memory efficiency.  Positional embeddings are
interpolated from the pre-trained 1024x1024.  Use set_resolution() for inference.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from segment_anything import sam_model_registry

from adapters import IndustrialAdapter
from classifier_head import DefectClassifierHead, CLASS_VOCAB, NUM_CLASSES
from model_setup import (
    EMBED_DIM,
    BOTTLENECK_DIM,
    _ADAPTER_PREFIX,
    _make_patched_forward,
)


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
    train_resolution : int
        Training resolution (default 512).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        bottleneck_dim: int = BOTTLENECK_DIM,
        train_resolution: int = 512,
    ):
        super().__init__()
        self.device = device
        self.train_resolution = train_resolution

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

        # ── Interpolate positional embeddings for training resolution ──
        # SAM pos_embed shape: (1, H, W, C)  NOT (1, C, H, W)
        patch_size = 16
        new_spatial = train_resolution // patch_size

        orig_pos_embed = self.image_encoder.pos_embed  # (1, 64, 64, 768)
        new_pos_embed = F.interpolate(
            orig_pos_embed.permute(0, 3, 1, 2),  # -> (1, 768, 64, 64)
            size=(new_spatial, new_spatial),
            mode="bicubic",
            align_corners=False,
        ).permute(
            0, 2, 3, 1
        )  # -> (1, 32, 32, 768)
        self.image_encoder.pos_embed = nn.Parameter(new_pos_embed)

        if hasattr(self.image_encoder, "rel_pos_h"):
            self.image_encoder.rel_pos_h = nn.Parameter(
                F.interpolate(
                    self.image_encoder.rel_pos_h,
                    size=(new_spatial, 1),
                    mode="bicubic",
                    align_corners=False,
                )
            )
            self.image_encoder.rel_pos_w = nn.Parameter(
                F.interpolate(
                    self.image_encoder.rel_pos_w,
                    size=(1, new_spatial),
                    mode="bicubic",
                    align_corners=False,
                )
            )

        self.prompt_encoder.image_embedding_size = (new_spatial, new_spatial)
        self.prompt_encoder.input_image_size = (train_resolution, train_resolution)

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

    def set_resolution(self, resolution: int):
        """Switch to a different resolution (e.g. 1024 for inference)."""
        patch_size = 16
        new_spatial = resolution // patch_size

        self.image_encoder.pos_embed = nn.Parameter(
            F.interpolate(
                self.image_encoder.pos_embed.permute(0, 3, 1, 2),
                size=(new_spatial, new_spatial),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        )
        if hasattr(self.image_encoder, "rel_pos_h"):
            self.image_encoder.rel_pos_h = nn.Parameter(
                F.interpolate(
                    self.image_encoder.rel_pos_h,
                    size=(new_spatial, 1),
                    mode="bicubic",
                    align_corners=False,
                )
            )
            self.image_encoder.rel_pos_w = nn.Parameter(
                F.interpolate(
                    self.image_encoder.rel_pos_w,
                    size=(1, new_spatial),
                    mode="bicubic",
                    align_corners=False,
                )
            )
        self.prompt_encoder.image_embedding_size = (new_spatial, new_spatial)
        self.prompt_encoder.input_image_size = (resolution, resolution)
        self.train_resolution = resolution

    def forward(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jointly predict defect mask and class logits.

        Parameters
        ----------
        image : torch.Tensor  (B, 3, H, W)
        boxes : torch.Tensor  (B, 4)  or None

        Returns
        -------
        mask_logits : (B, 1, 256, 256)
        class_logits : (B, 8)
        """
        B = image.shape[0]

        if boxes is None:
            H, W = image.shape[2], image.shape[3]
            boxes = torch.tensor(
                [[0, 0, W - 1, H - 1]],
                device=self.device,
            ).repeat(B, 1)

        # ── Image encoder — gradient checkpointing saves VRAM ─────
        image_embeddings = checkpoint(
            self.image_encoder,
            image,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        self.prompt_encoder.input_image_size = (image.shape[2], image.shape[3])
        self.prompt_encoder._input_image_size = (image.shape[2], image.shape[3])
        image_pe = self.prompt_encoder.get_dense_pe()

        # ── Decoder loop (SAM's mask decoder assumes B=1) ─────────
        mask_logits_list, class_logits_list = [], []

        for i in range(B):
            img_emb_i = image_embeddings[i : i + 1]
            box_i = boxes[i : i + 1]

            sparse, dense = self.prompt_encoder(
                points=None,
                boxes=box_i,
                masks=None,
            )

            mask_i, _ = self.mask_decoder(
                image_embeddings=img_emb_i,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )

            iou_token = self._iou_token
            class_i = self.classifier_head(iou_token)

            mask_logits_list.append(mask_i)
            class_logits_list.append(class_i)

        return (torch.cat(mask_logits_list, dim=0), torch.cat(class_logits_list, dim=0))

    def get_trainable_params(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]


# ── Smoke test ──────────────────────────────────────────────────────


def _run_smoke_test():
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "sam_vit_b_01ec64.pth"

    model = IndustrialSAM(ckpt, device="cpu", train_resolution=512)
    model.eval()

    B = 2
    dummy = torch.randn(B, 3, 512, 512)
    boxes = torch.tensor(
        [[50, 50, 250, 250], [100, 100, 300, 300]], dtype=torch.float32
    )

    with torch.no_grad():
        mask_logits, class_logits = model(dummy, boxes=boxes)

    print(f"mask_logits:  {mask_logits.shape}  (expected (2, 1, 128, 128))")
    print(f"class_logits: {class_logits.shape}  (expected (2, 8))")
    assert mask_logits.shape == (2, 1, 128, 128)
    assert class_logits.shape == (2, 8)
    print("Smoke test OK (512x512).")

    adapter_params = sum(
        p.numel() for n, p in model.named_parameters() if _ADAPTER_PREFIX in n
    )
    head_params = sum(p.numel() for p in model.classifier_head.parameters())
    total_trainable = sum(p.numel() for p in model.get_trainable_params())
    print(f"Adapter params:  {adapter_params:>10,}")
    print(f"Classifier head: {head_params:>10,}")
    print(f"Total trainable: {total_trainable:>10,}")


if __name__ == "__main__":
    _run_smoke_test()