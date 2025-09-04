# Visual Search – Landmark Identification

This project implements a **visual search system** for identifying landmarks from images. It was developed as part of a computer vision assignment using a dataset adapted from the Google Landmarks Dataset.

---

## 📊 Project Description

The goal is to determine whether two images belong to the same landmark or, given a query image, retrieve the **k most similar images** from a reference set.

The dataset contains **100 images across 20 landmarks** (e.g., Sydney Opera House, Golden Gate Bridge) with variations in scale, pose, and lighting.

Two main tasks were completed:

### Task 1 – Visual Search with Pretrained CNN

* Compute embeddings for query and reference images using a pretrained CNN.
* Measure similarity with a distance metric.
* Rank the k most similar images.
* Compute retrieval performance metrics.

### Task 2 – Learning an Embedding

* Choose a CNN architecture (same as Task 1).
* Train with **contrastive loss** or **triplet loss** using the *PyTorch Metric Learning* library.
* Evaluate retrieval results and compare against Task 1.

---

## 🛠️ Requirements

* Python ≥ 3.9
* PyTorch
* torchvision
* pytorch-metric-learning
* matplotlib
* numpy
* scikit-learn
* jupyter

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Clone the repository and open the notebooks:

```bash
git clone https://github.com/your-username/VisualSearch_Landmarks.git
cd VisualSearch_Landmarks
```

* `task1.ipynb` – Visual search using pretrained CNN embeddings
* `task2.ipynb` – Train custom embeddings with metric learning losses

Run the notebooks in order to reproduce results.

---

## 📈 Results

* Retrieval performance with pretrained CNN embeddings
* Improved results after training custom embeddings with contrastive/triplet loss
* Comparison of metrics between Task 1 and Task 2

---

## 📂 Files

* `task1.ipynb` – Implementation of Task 1
* `task2.ipynb` – Implementation of Task 2
* `utils.py` – Utility functions used in Task 1 and Task 2
* `requirements.txt` – Dependencies
* `README.md` – Project documentation

