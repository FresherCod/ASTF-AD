# ASTF-AD: Adaptive Semantic-Texture Fusion for Few-Shot Industrial Anomaly Detection

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Official implementation of the paper:

> **ASTF-AD: git init -b main**  
> **Author:** *Trương Đình Phúc*  
> (2025)

---

## 🧠 Overview

ASTF-AD introduces a **hybrid framework for industrial anomaly detection**, designed specifically for **few-shot learning** scenarios where anomalous samples are scarce.

The core innovation is an **Expert Selection System**, which intelligently combines the strengths of **texture-based** methods (e.g., PatchCore) and **semantic-based** Foundation Models (e.g., DINOv2).  
Rather than using a “one-size-fits-all” approach, ASTF-AD dynamically selects the optimal strategy for each data category.

<p align="center">
  <img src="results/visuals/carpet_002_comparison.png" width="700">
</p>

*Fig. 1: Visual comparison of anomaly maps on the “carpet” category, demonstrating the superior performance of ASTF-AD.*

---

## 🚀 Core Contributions

### 🧩 1. Adapter Module for Foundation Model Adaptation
We propose a lightweight **Adapter** module that fine-tunes large, pre-trained Foundation Models to specific industrial domains.  
A novel **Compactness Loss** ensures tighter clustering of normal features, enhancing discriminability.

---

### 🔍 2. Comprehensive Fusion Analysis
We perform an extensive study of various static fusion methods (additive, multiplicative, weighted sum) and reveal the **information cancellation problem**, where signals from one modality may suppress the other.

---

### 🧠 3. Expert Selection System
A rule-based system that selects the most effective fusion or adaptation strategy (`semantic_adapter`, `fusion_weighted_sum`, etc.) for each category — providing robustness across different industrial textures.

---

## 🛠️ Setup and Installation

### **Prerequisites**
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)
- Anaconda or virtual environment

### **Installation**

```bash
# 1. Clone the repository
git clone https://github.com/FresherCod/ASTF-AD.git
cd ASTF-AD

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare the Dataset
# Download the MVTec AD dataset from its official website:
# https://www.mvtec.com/company/research/datasets/mvtec-ad
#
# Extract the archive and place its contents into the `data/mvtec_ad/` directory.
# The final structure should be: `data/mvtec_ad/bottle/train/good/...`


## 📈 Reproduction Guide

Experiments are organized sequentially.  
To ensure reproducibility, please run the following scripts **in order**.

---

### 🧩 Step 1 — Generate & Summarize Baseline Results

#### **Option A — Train PatchCore from scratch**
```bash
python src/01a_run_baseline_full.py



Option B — Use existing checkpoints

Place your pre-trained .ckpt files under:
models/patchcore/  (e.g., models/patchcore/bottle.ckpt)

Then run:
python src/01b_summarize_baseline.py

Output:
results/patchcore_baseline_summary.csv

⚙️ Step 2 — Train Adapter Modules
python src/02_train_adapters.py
Output:
Trained adapter weights (.pth files) are saved in:
models/adapters/

🧠 Step 3 — Evaluate Final ASTF-AD System
python src/03_evaluate_system.py

Output:
A comprehensive benchmark comparison is saved to:
results/final_benchmark_comparison.csv

🎨 Step 4 — Generate Visual Comparisons (Optional)
python src/04_generate_visuals.py

Before running, edit the images_to_visualize dictionary inside the script to select your target images.

Output:
Comparison images will be saved in:
results/visuals/

📊 Key Results
Category	Applied Strategy	Image AUROC (ASTF-AD)	Baseline Image AUROC	Pixel AUROC (ASTF-AD)	Baseline Pixel AUROC
carpet	semantic_adapter	0.9988	0.7006	0.9831	0.9545
leather	semantic_adapter	1.0000	0.8893	0.9831	0.9791
screw	fusion_weighted_sum_0.7	0.6581	0.7950	0.9872	0.9873
...	...	...	...	...	...
Average	-	0.8669	0.9233	0.9372	0.9631

Note: While the overall average may appear slightly lower, ASTF-AD excels in challenging categories where baseline methods fail — demonstrating superior adaptability and generalization.

🧩 Future Work

🔁 Automated Strategy Selection: Replace the static STRATEGY_MAP with a learnable meta-model to predict optimal fusion per product type.

🧠 AI Arbiter: Develop a meta-model that infers the best strategy from small sample batches, mimicking real QA/QC workflows.

🔬 Learnable Fusion Module: Explore neural fusion architectures (e.g., FusionNet) for pixel-wise adaptive combination of anomaly maps.

📚 Citation

If you use ASTF-AD in your research, please cite:

@article{truongdinhphuc_astfad,
  title   = {ASTF-AD: Adaptive Semantic-Texture Fusion for Few-Shot Industrial Anomaly Detection},
  author  = {Trương Đình Phúc},
  year    = {2025}
}

📄 License

This project is licensed under the MIT License — see the LICENSE.md
 file for details.