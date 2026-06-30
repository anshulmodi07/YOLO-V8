# 🛰️ ISS Guardian — Space Station Object Detection

> Submission for the **Duality AI – Space Station Hackathon**

ISS Guardian is a highly optimized YOLOv8 model for detecting critical safety equipment (fire extinguishers, toolboxes, oxygen tanks) in a simulated space station environment, paired with a Flask web app for real-world demonstration.

**🏆 Final mAP@50: 88.6%**

---

## Table of Contents

- [Quick Start](#quick-start)
- [Environment & Dependencies](#environment--dependencies)
- [Running the Web App](#running-the-web-app)
- [Reproducing Final Results](#reproducing-final-results)
- [Understanding the Outputs](#understanding-the-outputs)
- [Optimization Methodology](#optimization-methodology)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)

---

## Quick Start

```bash
conda create -n duality_hackathon python=3.10 -y
conda activate duality_hackathon
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser and upload an image.

---

## Environment & Dependencies

| Requirement | Recommendation |
|---|---|
| Environment manager | Anaconda |
| Python version | 3.10 |
| GPU | CUDA-enabled GPU recommended for training (not required for inference) |

Key packages (full list in `requirements.txt`):

- `torch`, `torchvision` — PyTorch backbone
- `ultralytics` — YOLOv8
- `Flask` — web application server
- `Pillow`, `opencv-python-headless` — image processing

---

## Running the Web App

1. Activate your Conda environment and `cd` into the project directory.
2. Launch the server:
   ```bash
   python app.py
   ```
3. Navigate to `http://127.0.0.1:5000/`.
4. Click **"Choose an Image"**, upload a photo, and the app will run inference using `best.pt`, returning the original image side-by-side with predicted bounding boxes.

---

## Reproducing Final Results

To verify the reported **88.6% mAP@50**, run the prediction script against the included test set using the best model checkpoint:

```bash
python predict.py --model best.pt
```

This prints a results table to the terminal — check that the `mAP50` row for the `all` class matches **0.886**.

### (Optional) Re-running Training

```bash
python train.py
```

> ⚠️ Training is computationally intensive and requires a CUDA-enabled GPU. Expect a long runtime depending on hardware.

---

## Understanding the Outputs

**Web app:** returns the uploaded image plus an annotated copy with bounding boxes and confidence scores for `FireExtinguisher`, `ToolBox`, and `OxygenTank`.

**`predict.py`:** prints a metrics table to the terminal —

| Metric | Meaning |
|---|---|
| **mAP50** | Primary competition metric; overall detection accuracy at IoU 0.5 |
| **Precision** | Of all predictions made, the percentage that were correct |
| **Recall** | Of all real objects present, the percentage successfully detected |

A `confusion_matrix.png` is also saved to `runs/detect/val/`, showing class confusion and missed ("background") detections.

---

## Optimization Methodology

Our approach was a systematic, iterative process targeting specific model weaknesses at each stage:

1. **Baseline (`yolov8s`)** — established a starting point; revealed very low recall as the primary bottleneck.
2. **Boosting recall** — upgraded to `yolov8m`, increased training epochs, and added aggressive data augmentation. Recall improved substantially, but false positives ("hallucinations") rose in turn.
3. **Suppressing hallucinations** — introduced a custom loss configuration, raising the `kobj` parameter to more heavily penalize background false positives, cutting hallucinations while preserving recall gains.
4. **Final selection** — evaluated multiple checkpoints and selected the model with the best recall/precision balance, landing on our final **88.6% mAP@50**.

---

## Technology Stack

- **AI / ML:** PyTorch, Ultralytics YOLOv8, OpenCV, Pillow
- **Backend:** Python, Flask
- **Frontend:** HTML5, Tailwind CSS
- **Environment:** Conda

---

## Project Structure

```
.
├── app.py                  # Flask web application
├── train.py                # Final optimized training hyperparameters
├── predict.py               # CLI prediction/evaluation script
├── best.pt                 # Best-performing model weights (88.6% mAP)
├── templates/
│   └── index.html          # Web app frontend
├── requirements.txt        # Python dependencies
├── Hackathon_Report.pdf    # Final performance and analysis report
└── .gitignore
```
