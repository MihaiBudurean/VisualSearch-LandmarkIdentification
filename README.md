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

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 📈 Results

* Retrieval performance with pretrained CNN embeddings
* Improved results after training custom embeddings with contrastive/triplet loss


