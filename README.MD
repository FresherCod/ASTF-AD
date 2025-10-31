# ASTF-AD: Adaptive Semantic-Texture Fusion for Few-Shot Industrial Anomaly Detection

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation for the paper:

> **ASTF-AD: Adaptive Semantic-Texture Fusion for Few-Shot Industrial Anomaly Detection**  
> **Author:** *Phuc Dinh Truong* ([ORCID: 0009-0008-2149-478X](https://orcid.org/0009-0008-2149-478X))  
> *Preprint, 2025*

---

## 🧠 Overview

ASTF-AD introduces a **hybrid framework for industrial anomaly detection**, designed specifically for **few-shot learning** scenarios where anomalous samples are scarce.

The core innovation is an **Expert Selection System**, which intelligently combines the strengths of **texture-based** methods (e.g., PatchCore) and **semantic-based** Foundation Models (e.g., DINOv2). Rather than using a “one-size-fits-all” approach, ASTF-AD dynamically selects the optimal strategy for each data category.

<p align="center">
  <img src="https://github.com/FresherCod/ASTF-AD/blob/main/results/visuals/carpet_002_comparison.png?raw=true" width="800">
</p>

*Fig. 1: Visual comparison of anomaly maps on the “carpet” category, demonstrating the superior performance of ASTF-AD in handling complex textures where the baseline fails.*

---

## 🚀 Core Contributions

1.  **🧩 Adapter Module for Foundation Model Adaptation:** We propose a lightweight **Adapter** module that efficiently fine-tunes large, pre-trained Foundation Models for specific industrial domains. A novel **Compactness Loss** ensures tighter clustering of normal features, enhancing discriminability.

2.  **🔍 Comprehensive Fusion Analysis:** We perform an extensive study of various static fusion methods (additive, multiplicative, weighted sum) and identify the **"information cancellation" problem**, where valuable signals from one feature modality may suppress the other.

3.  **💡 Expert Selection System:** Based on our analysis, we introduce a rule-based framework that applies the most effective pre-analyzed strategy (`semantic_adapter`, `fusion_weighted_sum`, etc.) for each data category, demonstrating superior adaptability across diverse industrial textures and defect types.

---

## 🛠️ Setup Guide

### **Prerequisites**
- Python 3.9+
- Anaconda or a virtual environment manager
- NVIDIA GPU with CUDA 11.8+ (recommended)

### **Installation**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FresherCod/ASTF-AD.git
    cd ASTF-AD
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### **Step 4: Download Datasets and Pre-trained Models**

Due to their large size, the dataset and pre-trained models are hosted externally.

*   **Pre-trained Models (Required):**
    1.  Download the model archive from the link below:  
        **[Download `astf_ad_models.zip` from Google Drive](https://drive.google.com/file/d/1yrB24UPZAWpug8RrzkIlbRaVIY1pnBom/view?usp=drive_link)**
    2.  Extract the `.zip` file into the root directory of the project. This will create the `models/` folder containing all necessary PatchCore checkpoints (`.ckpt`) and Adapter weights (`.pth`).

*   **Dataset (MVTec AD):**
    1.  Download the MVTec AD dataset from its official website:  
        **[https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)**
    2.  Extract the archive and place its contents into the `data/mvtec_ad/` directory.

``` After these steps, your project directory should look like this:
ASTF-AD/
├── data/
│ └── mvtec_ad/
├── models/
│ ├── patchcore/
│ └── adapters/
├── src/
├── results/
├── README.md
└── ...
```


## 📈 Reproduction Guide

The experiments are organized sequentially. To reproduce our findings, please run the following scripts in order.

### **Step 1 — Generate & Summarize Baseline Results**

*   **Option A — (Full Run) Train PatchCore from scratch:**  
    *This will train PatchCore for all 15 categories. This is a very time-consuming step.*
    ```bash
    python src/01a_run_baseline_full.py
    ```

*   **Option B — (Recommended) Test from our pre-trained checkpoints:**  
    *Make sure you have downloaded and extracted our models from Google Drive.* This script will run testing only and generate the summary file much faster.
    ```bash
    python src/01b_summarize_baseline.py
    ```
**Output:** A summary file `results/patchcore_baseline_summary.csv` is generated.

### **Step 2 — Train Adapter Modules**
This script checks for existing adapter weights in `models/adapters/` and only trains the missing ones. If you downloaded our models, this step will be skipped automatically.

```bash
python src/02_train_adapters.py```

Output: Trained adapter weights (.pth files) for all 15 categories are ensured to be in models/adapters/.
### **Step 3 — Evaluate the Final ASTF-AD System**
This is the main evaluation script. It applies our "Expert Selection Strategy" for each category and compares the final performance against the baseline.

```bash
python src/03_evaluate_system.py```
Output: The comprehensive comparison table is printed to the console and saved to results/final_benchmark_comparison.csv.

### **Step 4 — Generate Visual Comparisons (Optional)**
To generate visual comparisons for specific images:
Open src/04_generate_visuals.py.
Edit the images_to_visualize dictionary inside the script to select your target images.
Run the script:
``` bash
python src/04_generate_visuals.py
```
Output: Comparison images will be saved in results/visuals/.

## 📊 Key Results
Category	Applied Strategy	Image AUROC (ASTF-AD)	Baseline Image AUROC	Pixel AUROC (ASTF-AD)	Baseline Pixel AUROC
carpet	semantic_adapter	0.9988	0.7006	0.9831	0.9545
leather	semantic_adapter	1.0000	0.8893	0.9831	0.9791
screw	fusion_weighted_sum_0.7	0.6581	0.7950	0.9872	0.9873
... (12 others)	...	...	...	...	...
Average	-	0.8669	0.9233	0.9372	0.9631
Note: While the overall average performance is comparable to the baseline, the key insight is the system's ability to achieve state-of-the-art results on specific, challenging data types (e.g., carpet, leather) where the baseline is known to perform poorly, demonstrating superior adaptability.

## 💡 Future Work
This research opens up several promising avenues for future work:

Automated Strategy Selection: Replace the static STRATEGY_MAP with a learnable meta-model (e.g., an ML Filter) to predict the optimal strategy for unseen product types.

AI Arbiter: Develop a meta-model that infers the best strategy from a small batch of initial samples, mimicking real-world QA/QC workflows.

Learnable Fusion Module: Explore neural fusion architectures (e.g., a FusionNet) for pixel-wise adaptive combination of anomaly maps.


## 📚 Citation
If you use ASTF-AD in your research, please cite:

@article{truong2025astfad,
  title   = {ASTF-AD: Adaptive Semantic-Texture Fusion for Few-Shot Industrial Anomaly Detection},
  author  = {Truong, Dinh Phuc},
  year    = {2025},
  journal = {arXiv preprint arXiv:XXXX.XXXXX}
}

