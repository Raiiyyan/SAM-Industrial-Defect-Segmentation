"""
DefectClassifierHead — lightweight MLP that maps SAM's 256-dim IoU token
to 8-class defect logits.

The IoU token is extracted from SAM's mask decoder after the TwoWayTransformer
has let it cross-attend to all image features.  It represents "what defect is
present" globally, making it the natural choice for classification.

Class vocabulary (8 classes, indices 0-7):
  0: Flawless (no defect)
  1: Surface Scratch
  2: Structural Crack
  3: Hole / Puncture
  4: Inclusion
  5: Missing Component
  6: Discoloration / Stain
  7: Geometric Deformation

**Important**: The head outputs **raw logits** (no softmax).  Feed them
directly to ``torch.nn.CrossEntropyLoss``.  Do NOT apply softmax before the
loss function — that would double-softmax and break training.
"""

import torch
import torch.nn as nn

# ── Class vocabulary ──────────────────────────────────────────────────

CLASS_VOCAB = {
    0: "Flawless",
    1: "Surface Scratch",
    2: "Structural Crack",
    3: "Hole / Puncture",
    4: "Inclusion",
    5: "Missing Component",
    6: "Discoloration / Stain",
    7: "Geometric Deformation",
}

NUM_CLASSES = len(CLASS_VOCAB)


class DefectClassifierHead(nn.Module):
    """Map a 256-dim IoU token to 8-class logits.

    Parameters
    ----------
    input_dim : int
        Token dimension (256 for SAM ViT-B mask decoder).
    num_classes : int
        Number of defect classes (default 8).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(self, input_dim: int = 256, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no softmax applied).

        Parameters
        ----------
        x : torch.Tensor  shape (B, 256)  — IoU token from mask decoder.

        Returns
        -------
        torch.Tensor  shape (B, 8) — raw class logits.
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
