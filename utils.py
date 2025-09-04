import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for single GPU
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Dataset Utilities ----------

class LandmarkDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.image_paths[idx]

def load_dataset(image_dir, class_json, transform):
    with open(class_json, "r") as f:
        class_map = {str(entry["id"]): entry["landmark"] for entry in json.load(f)}

    image_paths = []
    labels = []

    for class_id, class_name in class_map.items():
        folder = os.path.join(image_dir, class_id)
        if not os.path.isdir(folder):
            continue
        for img_file in os.listdir(folder):
            if img_file.endswith(".jpg") or img_file.endswith(".jpeg") or img_file.endswith(".png"):
                image_paths.append(os.path.join(folder, img_file))
                labels.append(class_id)

    # Simple query/reference split: last image per class as query
    query_paths, query_labels = [], []
    reference_paths, reference_labels = [], []

    seen = set()
    for path, label in zip(image_paths, labels):
        if label not in seen:
            query_paths.append(path)
            query_labels.append(label)
            seen.add(label)
        else:
            reference_paths.append(path)
            reference_labels.append(label)

    return {
        "query": LandmarkDataset(query_paths, query_labels, transform),
        "reference": LandmarkDataset(reference_paths, reference_labels, transform)
    }

# ---------- Model Utilities ----------

def get_embedding_model(arch="resnet50", pretrained=True, output_dim=2048):
    model = models.__dict__[arch](pretrained=pretrained)
    if hasattr(model, "fc"):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError("Unknown architecture")

    return model

# ---------- Embedding Extraction ----------

def extract_embeddings(model, dataset, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    labels = []
    paths = []

    with torch.no_grad():
        for images, lbls, img_paths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            out = model(images).cpu()
            embeddings.append(out)
            labels.extend(lbls)
            paths.extend(img_paths)

    return {
        "embeddings": torch.cat(embeddings),
        "labels": labels,
        "paths": paths
    }

# ---------- Retrieval and Evaluation ----------

def compute_similarity(query_embs, ref_embs, metric="cosine"):
    if metric == "cosine":
        query_norm = torch.nn.functional.normalize(query_embs, p=2, dim=1)
        ref_norm = torch.nn.functional.normalize(ref_embs, p=2, dim=1)
        return torch.matmul(query_norm, ref_norm.T)
    elif metric == "euclidean":
        return -torch.cdist(query_embs, ref_embs)
    else:
        raise ValueError("Unsupported metric")

def retrieve_top_k(query_data, ref_data, k=5, metric="cosine"):
    similarities = compute_similarity(query_data["embeddings"], ref_data["embeddings"], metric)
    top_k_indices = torch.topk(similarities, k=k, dim=1).indices
    top_k_labels = []

    for indices in top_k_indices:
        top_k_labels.append([ref_data["labels"][i] for i in indices.tolist()])

    return {
        "predictions": top_k_labels,
        "ground_truth": query_data["labels"]
    }

def evaluate_retrieval(retrieval_result, k=5):
    y_true = retrieval_result["ground_truth"]
    y_pred = retrieval_result["predictions"]

    precision_scores = []
    recall_scores = []
    average_precisions = []

    for true_label, preds in zip(y_true, y_pred):
        top_k_preds = preds[:k]
        correct_preds = [1 if pred == true_label else 0 for pred in top_k_preds]
        num_relevant = sum([label == true_label for label in preds])

        # Precision@k
        precision = sum(correct_preds) / k
        precision_scores.append(precision)

        # Recall@k (how much of relevant retrieved in top-k)
        # Since 1 query per class, max recall is 1 if at least one match found
        recall = 1.0 if true_label in top_k_preds else 0.0
        recall_scores.append(recall)

        # AP@k (Average Precision)
        hits = 0
        cumulative_precision = 0.0
        for idx, pred in enumerate(top_k_preds):
            if pred == true_label:
                hits += 1
                cumulative_precision += hits / (idx + 1)
        ap = cumulative_precision / hits if hits > 0 else 0.0
        average_precisions.append(ap)

    precision_at_k = sum(precision_scores) / len(precision_scores)
    recall_at_k = sum(recall_scores) / len(recall_scores)
    mean_ap = sum(average_precisions) / len(average_precisions)

    return {
        f"Precision@{k}": round(precision_at_k, 4),
        f"Recall@{k}": round(recall_at_k, 4),
        f"mAP@{k}": round(mean_ap, 4)
    }

# ---------- Task2 Utils ----------

class EmbeddingNet(nn.Module):
    def __init__(self, base_arch="resnet50", embed_dim=128):
        super().__init__()
        base = get_embedding_model(base_arch, pretrained=True)
        self.feature_extractor = base
        self.embedding = nn.Linear(2048, embed_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.embedding(x)
        return x


def train_or_load_model(model, model_path, loss_fn, miner, sampler, train_dataset, device, epochs=10):

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Training model and saving to {model_path}...")
        loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            epoch_loss = 0
            for images, labels, _ in loader:
                images, labels = images.to(device), torch.tensor(labels).to(device)
                optimizer.zero_grad()
                embeddings = model(images)
                mined = miner(embeddings, labels)
                loss = loss_fn(embeddings, labels, mined)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(loader):.4f}")

        torch.save(model.state_dict(), model_path)
        model.eval()

    return model
