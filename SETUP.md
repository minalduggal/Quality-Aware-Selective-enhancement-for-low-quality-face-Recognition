# Project Setup Guide

## Quality-Aware Selective Enhancement for Low-Quality Face Recognition

### 1. Prerequisites

- Python 3.14.2 (current development environment)
- Git
- VS Code
- TinyFace dataset (required for benchmark evaluation)

### 2. Environment Setup

Open the terminal inside the project directory.

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 3. Project Modules

| Folder | Purpose |
|---|---|
| RetinaFace | Face detection and cropping |
| SHQS | Face quality assessment and enhancement selection |
| ArcFace | Feature extraction and similarity matching |
| Integration | Pipeline integration |

### 4. Proposed Pipeline

Input Image → RetinaFace → SHQS → Selective Enhancement → ArcFace → Cosine Similarity → Identity Ranking

Enhancement routing:

- NO: Original image
- SOFT: Adaptive CLAHE
- STRONG: CodeFormer

### 5. Dataset Configuration

Download and extract the TinyFace dataset separately.

Set the dataset path according to your system:

```powershell
$env:TINYFACE_ROOT = "YOUR_TINYFACE_DATASET_PATH"
```

This variable must be configured in each new terminal session before running dataset-related scripts.

### 6. Run TinyFace Baseline

```powershell
python ArcFace/tinyface_full_evaluation.py
```

### 7. Test Integration

Create an `images` folder in the project directory and place an authorized test image named `test.jpg` inside it.

```powershell
python Integration/pipeline.py images/test.jpg
```

### 8. Pending Work

- CodeFormer integration using its existing environment.
- RetinaFace inference testing.
- ArcFace connection with the integrated pipeline.
- End-to-end pipeline testing.
- Comparison of baseline, unconditional enhancement and selective enhancement.

### 9. Important Notes

- Dataset, model weights, virtual environments and generated images are not included in the repository.
- RetinaFace import has been verified, but actual inference testing is pending.
- TensorFlow is not installed in the current environment. Check whether it is required during inference.
- CodeFormer dependencies should be maintained separately until compatibility is verified.
- The requirements file represents the current development environment. Fresh installation on another system has not yet been verified.