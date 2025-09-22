# CReFT-CAD: Boosting Orthographic Projection Reasoning for CAD via Reinforcement Fine-Tuning

This is the **official repository** for the paper:

> **CReFT-CAD: Boosting Orthographic Projection Reasoning for CAD via Reinforcement Fine-Tuning**

📌 **All code, pretrained models, and datasets will be released upon paper acceptance.**

The dataset can be accessed here: [Dataset](https://www.modelscope.cn/datasets/zhuofanChen/TriView2CAD)
The pretrained models can be accessed here: [Model]([https://www.modelscope.cn/datasets/zhuofanChen/TriView2CAD](https://www.modelscope.cn/models/zhuofanChen/CReFT-CAD))

# TriView2CAD Dataset

**TriView2CAD** is the first large-scale, open-source benchmark specifically designed for orthographic projection reasoning in Computer-Aided Design (CAD). It supports tasks ranging from engineering drawing interpretation to 3D geometry reconstruction, and serves as a foundation for training and evaluating multi-modal models in industrial scenarios.

## 📦 Dataset Overview

TriView2CAD consists of:
- **200,000 synthetic samples** for training and in-domain evaluation;

Each sample is annotated with **precise dimension labels** and includes **six interoperable data modalities**:
1. **Parameter Tables** (JSON)
2. **Vector CAD Drawings** (DXF)
3. **Raster Images** (PNG; front, top, side views)
4. **Executable CAD Scripts**
5. **STEP Files** (3D models)
6. **B-Rep Files** (3D models)

This rich format enables seamless support for tasks across design, simulation, and manufacturing workflows.

## ⚙️ Dataset Construction

We focus on **prefabricated bridge piers**, a modular structure commonly found in civil engineering. A constraint-guided pipeline ensures the geometric and structural validity of each sample:

### Step 1: Constraint-Guided Parameter Sampling
- A 15-dimensional parameter space is defined.
- **Intra-view constraints** ensure physical validity and topology closure.
- **Inter-view constraints** guarantee consistency across orthographic views (height, width, depth).

### Step 2: 2D Drawing & Raster Image Generation
- The `ezdxf` library converts sampled parameters into vectorized DXF drawings.
- Geometric primitives (lines, circles, arcs) are organized into semantic layers.
- High-resolution raster images (front, top, side) are rendered in FreeCAD.

### Step 3: 3D Model Generation
- Using FreeCAD's scripting interface, 3D models are programmatically reconstructed.
- Both **STEP** and **B-Rep** files are generated to support CAD simulation and interoperability.

## 🧠 Supported Reasoning Tasks

TriView2CAD enables evaluation of complex reasoning capabilities via three key task categories:

1. **Dimension Recognition and Mapping**  
   Identify each annotated dimension and match it to its geometric feature.

2. **Primitive Counting**  
   Count the number of specified components (e.g., pier columns, pile bases).

3. **Composite Parameter Computation**  
   Derive engineering-critical values based on multiple parameters (e.g., spacing = width + gap).

A total of **15 parameters** are evaluated per sample, including:
- 6 recognition attributes
- 3 counting attributes
- 6 computed attributes

Evaluation accuracy is computed per parameter and averaged across the test set.

## 📊 Benchmarking & Generalization

TriView2CAD has been used to benchmark seven leading vision–language models. Our experiments demonstrate:
- Significant performance gaps between pretrained VLMs and reasoning-tuned models.
- Real-world samples introduce realistic noise and complexity (e.g., overlapping annotations, occlusions).
- Chain-of-Thought (CoT) and curriculum-based fine-tuning dramatically improve reasoning accuracy and OOD robustness.
