# Laptop Setup Prompt

Copy-paste the entire block below into a fresh opencode session on the laptop.

---

I am setting up the SAM-Industrial-Defect-Segmentation project. Do NOT modify any existing .py files.

1. Ask me which LETTER drive the dataset folder is on (e.g. D, E, F).
2. Set the variable `$DRIVE` to that letter.
3. Set `$PROOT` as the project root = `$DRIVE:/SAM-Industrial-Defect-Segmentation`.
4. Set `$VENV_DIR` = `$DRIVE:/venvs/sam-venv`. Ask me first, but suggest this default.
5. Create directory `$VENV_DIR` if it doesn't exist.
6. Create a Python virtual environment at `$VENV_DIR`:
   ```
   python -m venv $VENV_DIR
   ```
7. Activate it and install the laptop requirements (NOT the regular requirements.txt):
   ```
   $VENV_DIR/Scripts/Activate.ps1
   pip install -r $PROOT/requirements_laptop.txt --index-url https://download.pytorch.org/whl/cu124
   ```
8. Check if `$PROOT/sam_vit_b_01ec64.pth` exists:
   - If NOT: download it from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` and place it in `$PROOT`.
   - If YES: skip.
9. Install the `segment-anything` package from pip (it's in requirements_laptop.txt but ensure it's installed):
   ```
   pip install segment-anything
   ```
10. Run training with the laptop-optimised script. First ask me if I want to:
    - Change the number of epochs (default 50)
    - Change batch size (default 1)
    - Enable/disable AMP (default on)
    - Change gradient accumulation (default 4)
    Then run:
    ```
    cd $PROOT
    $VENV_DIR/Scripts/python.exe train_laptop.py --root-dir $DRIVE:/Dataset --epochs 50 --batch-size 1 --grad-accum 4 --amp
    ```

Keep me updated on each step. If a step fails, stop and tell me what happened.
