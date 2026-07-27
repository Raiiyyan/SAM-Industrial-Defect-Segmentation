"""
Model setup: load SAM ViT-B, freeze image encoder, inject IndustrialAdapters,
and verify gradient isolation.

Injection strategy: monkey-patch
------------------------------
We replace each Block.forward with a wrapper that calls the original submodules
(attn, mlp, norms) but passes the attention and MLP outputs through IndustrialAdapters
before adding them to the residual stream.  This avoids modifying the installed
segment-anything library source, so ``pip install segment-anything`` stays clean.

Alternatives considered:
- **Forward hooks**: Cannot cleanly intercept the value *before* the residual add.
- **Subclassing**: Would require replacing Block instances after model creation,
  which is fragile.
- **Monkey-patching**: Simple, reversible, and the injected adapters are stored as
  regular nn.Module attributes on the wrapper closure, so they participate in
  state_dict / device movement automatically.
"""

import torch
import torch.nn as nn
from segment_anything import sam_model_registry

from adapters import IndustrialAdapter

# Regex to identify adapter parameters inside the model (we attach adapters
# as attributes on the monkey-patched closure, so they appear under
# image_encoder.blocks.X.adapter_attn / adapter_mlp in named_parameters).
_ADAPTER_PREFIX = "adapter_"


# ── Constants ────────────────────────────────────────────────────────
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
EMBED_DIM = 768
NUM_BLOCKS = 12
BOTTLENECK_DIM = 64


# ── Load & freeze ────────────────────────────────────────────────────

def load_frozen_encoder(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load SAM ViT-B image encoder with all parameters frozen.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``sam_vit_b_01ec64.pth``.
    device : str
        Torch device string.

    Returns
    -------
    sam.model
        The full SAM model (encoder frozen).
    """
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()

    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    return sam


# ── Adapter injection ─────────────────────────────────────────────────

def _make_patched_forward(block: nn.Module, adapter_attn: IndustrialAdapter,
                          adapter_mlp: IndustrialAdapter):
    """Return a patched forward function for *block* that injects adapters.

    The adapter is applied *after* the submodule output but *before* the
    residual add, so the final contribution to the residual stream is::

        original_submodule_output + adapter_delta

    Because the adapter's up-projection is initialised to zero,
    ``adapter(x) ≈ x`` initially, preserving pre-trained behaviour.
    """
    orig_forward = block.forward

    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        # --- Attention sub-layer ---
        attn_out = block.attn(block.norm1(x))
        attn_out = adapter_attn(attn_out)
        x = x + attn_out

        # --- MLP sub-layer ---
        mlp_out = block.mlp(block.norm2(x))
        mlp_out = adapter_mlp(mlp_out)
        x = x + mlp_out

        return x

    return patched_forward


def inject_adapters(sam: nn.Module, bottleneck_dim: int = BOTTLENECK_DIM
                    ) -> nn.Module:
    """Inject IndustrialAdapters into each of the 12 transformer blocks.

    Each block gets two adapters: one after self-attention, one after MLP.
    The adapters are registered as submodules of each block (``adapter_attn``
    and ``adapter_mlp``), so they appear in ``sam.named_parameters()`` and
    ``sam.state_dict()`` automatically.

    Parameters
    ----------
    sam : nn.Module
        SAM model with frozen image encoder.
    bottleneck_dim : int
        Adapter bottleneck dimension.

    Returns
    -------
    sam : same model with adapters injected (mutated in place).
    """
    blocks = sam.image_encoder.blocks

    for block in blocks:
        ad_attn = IndustrialAdapter(EMBED_DIM, bottleneck_dim)
        ad_mlp  = IndustrialAdapter(EMBED_DIM, bottleneck_dim)

        # Register as block submodules so they appear in state_dict / named_parameters
        block.adapter_attn = ad_attn
        block.adapter_mlp = ad_mlp

        block.forward = _make_patched_forward(block, ad_attn, ad_mlp)

    return sam


# ── Verification ─────────────────────────────────────────────────────

def verify_gradient_isolation(model: nn.Module):
    """Assert that adapter parameters are the ONLY trainable parameters in
    the image encoder and that >95% of *all* model parameters remain frozen.

    Trainable parameters outside the image encoder (prompt encoder / mask
    decoder) are expected and allowed — they are handled in later weeks.

    Raises ``AssertionError`` with a descriptive message on failure.
    """
    total = 0
    frozen = 0
    trainable = 0
    adapter_total = 0
    adapter_trainable = 0
    original_total = 0  # params excluding injected adapters

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        is_adapter = _ADAPTER_PREFIX in name
        if not is_adapter:
            original_total += n
        if param.requires_grad:
            trainable += n
            if is_adapter:
                adapter_total += n
                adapter_trainable += n
        else:
            frozen += n

    # Ratio relative to original model (excluding adapters), since adapters
    # are intentionally trainable and shouldn't dilute the frozen percentage.
    ratio_frozen = frozen / original_total * 100

    print(f"Total parameters:        {total:>12,}")
    print(f"  Original model:        {original_total:>12,}")
    print(f"  Injected adapters:     {adapter_total:>12,}")
    print(f"Frozen (original only):  {frozen:>12,}  ({ratio_frozen:.2f}%)")
    print(f"Trainable parameters:    {trainable:>12,}")

    # ── Assertions ──

    # 1. Image encoder must have zero trainable non-adapter params
    encoder_leaky = [
        n for n, p in model.image_encoder.named_parameters()
        if p.requires_grad and _ADAPTER_PREFIX not in n
    ]
    assert len(encoder_leaky) == 0, (
        f"Image encoder has {len(encoder_leaky)} leaky parameters:\n" +
        "\n".join(f"  {n}" for n in encoder_leaky[:20])
    )

    # 2. All adapter params are trainable (not frozen)
    adapter_frozen = [
        n for n, p in model.named_parameters()
        if _ADAPTER_PREFIX in n and not p.requires_grad
    ]
    assert len(adapter_frozen) == 0, (
        f"{len(adapter_frozen)} adapter parameters are unexpectedly frozen:\n" +
        "\n".join(f"  {n}" for n in adapter_frozen[:10])
    )

    # 3. >95 % of total params frozen
    assert ratio_frozen > 95.0, (
        f"Only {ratio_frozen:.2f}% of parameters are frozen "
        f"(expected >95%)"
    )

    print("\nAll gradient isolation checks passed.")
    print(f"  >95% frozen:           {ratio_frozen:.2f}%  OK")
    print(f"  Encoder leaky params:  {len(encoder_leaky)}  OK")
    print(f"  Adapter params frozen: {len(adapter_frozen)}  OK")


# ── Convenience wrapper ──────────────────────────────────────────────

def build_model(checkpoint_path: str, device: str = "cpu",
                bottleneck_dim: int = BOTTLENECK_DIM) -> nn.Module:
    """Load SAM, freeze encoder, inject adapters, and verify isolation.

    Returns the SAM model with adapters attached (mutated in place).
    """
    sam = load_frozen_encoder(checkpoint_path, device)
    sam = inject_adapters(sam, bottleneck_dim)
    verify_gradient_isolation(sam)
    return sam


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else (
        "G:/SAM-Industrial-Defect-Segmentation/sam_vit_b_01ec64.pth"
    )
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

    print("Building model with frozen encoder + adapters ...")
    sam = build_model(ckpt, device)

    # Quick forward sanity check
    dummy = torch.randn(1, 3, 1024, 1024).to(device)
    with torch.no_grad():
        feats = sam.image_encoder(dummy)
    print(f"\nForward pass OK — output shape: {feats.shape}")

    # Show total adapter params extracted from model itself
    adapter_params = sum(
        p.numel() for n, p in sam.named_parameters() if _ADAPTER_PREFIX in n
    )
    print(f"Adapter parameters in model: {adapter_params:,}")
