# Local SAM 3 checkpoint

Place the official SAM 3 image checkpoint here as **sam.pt**.
The original filename **sam3.pt** is also supported as a fallback. If both exist,
`sam.pt` takes precedence. Existing user weights are never renamed or moved.

Weights are not downloaded automatically. The BAT verifies CUDA and the local
checkpoint before opening the application. Renaming a different model's weights
does not make them compatible with SAM 3.

Official model source: https://huggingface.co/facebook/sam3

Sign in to Hugging Face, review and accept the model conditions, request access,
and wait for approval if required. Open **Files and versions** and download
**sam3.pt** (about 3.45 GB). Copy the completed download into this folder as
`sam3.pt` or `sam.pt`. Do not use `model.safetensors` or SAM 2 weights.

Run `install.bat` from the project folder for dependency installation and a
final checklist, including these download instructions when the model is missing.
