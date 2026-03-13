"""
Custom face recognizer using PCA (Eigenfaces) + cosine similarity.
Works with standard opencv-python — no opencv-contrib required.

The model is saved/loaded as a .npz file.
"""
import cv2
import numpy as np
import os
import pickle
from typing import Optional

from config import TRAINER_FILE, DATASET_DIR


TRAINER_NPZ = TRAINER_FILE.replace(".yml", ".npz")
FACE_SIZE = (100, 100)  # normalize all faces to this size
N_COMPONENTS = 50       # PCA components to keep


class EigenFaceRecognizer:
    """Simple PCA-based face recognizer."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None  # (n_components, n_pixels)
        self.projections: Optional[np.ndarray] = None  # (n_samples, n_components)
        self.labels: list[int] = []
        self.trained = False

    # ──────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────

    def train(self, faces: list[np.ndarray], labels: list[int]):
        """
        Train on a list of grayscale face images (any size) and integer labels.
        """
        data = np.array([self._preprocess(f) for f in faces], dtype=np.float64)
        self.labels = labels

        # PCA
        self.mean = data.mean(axis=0)
        centered = data - self.mean

        # Use SVD for efficient PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(N_COMPONENTS, len(faces) - 1, Vt.shape[0])
        self.components = Vt[:k]            # (k, n_pixels)
        self.projections = centered @ self.components.T  # (n_samples, k)
        self.trained = True

    def predict(self, face: np.ndarray) -> tuple[int, float]:
        """
        Predict label for a face image.
        Returns (label, distance) where lower distance = better match.
        """
        if not self.trained:
            return -1, 9999.0

        vec = self._preprocess(face) - self.mean
        proj = vec @ self.components.T  # (k,)

        # Cosine distances to all training projections
        dists = self._cosine_distances(proj, self.projections)
        best_idx = int(np.argmin(dists))
        return self.labels[best_idx], float(dists[best_idx])

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save(self, path: str = TRAINER_NPZ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
            projections=self.projections,
            labels=np.array(self.labels),
        )

    def load(self, path: str = TRAINER_NPZ) -> bool:
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            self.mean = data["mean"]
            self.components = data["components"]
            self.projections = data["projections"]
            self.labels = data["labels"].tolist()
            self.trained = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(face: np.ndarray) -> np.ndarray:
        """Resize, equalize, and flatten a face image."""
        resized = cv2.resize(face, FACE_SIZE)
        equalized = cv2.equalizeHist(resized)
        return equalized.flatten().astype(np.float64)

    @staticmethod
    def _cosine_distances(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine distance from vec to each row of matrix."""
        norm_vec = np.linalg.norm(vec)
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > 0
        distances = np.ones(len(matrix))
        if norm_vec > 0:
            dots = matrix[valid] @ vec
            distances[valid] = 1.0 - dots / (norms[valid] * norm_vec)
        return distances


# ── Module-level singleton ────────────────────────────────────

_recognizer: Optional[EigenFaceRecognizer] = None


def get_recognizer() -> EigenFaceRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = EigenFaceRecognizer()
    return _recognizer


def model_file_path() -> str:
    return TRAINER_NPZ