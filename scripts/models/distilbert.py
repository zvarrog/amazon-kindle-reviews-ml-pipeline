from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModel, AutoTokenizer

from scripts.config import DISTILBERT_EARLY_STOP_PATIENCE
from scripts.logging_config import get_logger

log = get_logger(__name__)


class DistilBertClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-совместимый классификатор на базе DistilBERT.

    Реализует интерфейс sklearn estimator для использования в Pipeline.
    Обучает только classification head, замораживая веса base model.
    """

    def __init__(
        self,
        epochs: int = 1,
        lr: float = 1e-4,
        max_len: int = 128,
        batch_size: int = 8,
        seed: int = 42,
        device: str | None = None,
    ):
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self._fitted = False
        self._tokenizer = None
        self._base_model = None
        self._head = None
        self._device_actual = None
        self.classes_ = None  # sklearn convention: публичный атрибут

    def _tokenize(self, texts, return_tensors: str = "pt"):
        if self._tokenizer is None:
            raise RuntimeError("Токенизатор не инициализирован")
        return self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors=return_tensors,
        )

    def _get_device(self):
        if self._head is not None:
            try:
                return next(self._head.parameters()).device
            except (StopIteration, RuntimeError) as e:
                log.debug("Не удалось получить device из head: %s", e)
        if self._base_model is not None:
            try:
                return next(self._base_model.parameters()).device
            except (StopIteration, RuntimeError) as e:
                log.debug("Не удалось получить device из base_model: %s", e)
        return self._device_actual or torch.device("cpu")

    def _extract_texts(self, x: Any) -> list[str]:
        """Безопасно извлекает тексты из входных данных разных типов."""
        if isinstance(x, pd.DataFrame):
            # Если передана таблица, ищем колонку с текстом
            col = "reviewText"
            if col not in x.columns:
                # Если такой колонки нет, берем первую строковую колонку
                str_cols = x.select_dtypes(include=["object"]).columns
                if not str_cols.empty:
                    col = str_cols[0]
                    log.warning("Колонка 'reviewText' не найдена, используем '%s'", col)
                else:
                    raise ValueError(
                        f"DataFrame не содержит колонку '{col}' или другие текстовые поля"
                    )
            return list(x[col].fillna("").astype(str).tolist())

        if isinstance(x, pd.Series):
            return list(x.fillna("").astype(str).tolist())

        if isinstance(x, np.ndarray):
            # Если это 2D массив, берем первый столбец (предполагаем, что текст там)
            if x.ndim > 1:
                return list(x[:, 0].astype(str).tolist())
            return list(x.astype(str).tolist())

        if isinstance(x, list):
            return [str(i) if i is not None else "" for i in x]

        return [str(x)]

    def fit(self, x, y, x_val=None, y_val=None):
        """Обучает модель на текстах."""
        torch.manual_seed(self.seed)
        texts = self._extract_texts(x)

        self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self._base_model = AutoModel.from_pretrained("distilbert-base-uncased")

        device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Обучение DistilBERT на устройстве: %s", device_str)

        device = torch.device(device_str)
        for p in self._base_model.parameters():
            p.requires_grad = False

        unique_labels = np.unique(y)
        self.classes_ = unique_labels
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}

        hidden = self._base_model.config.hidden_size
        n_classes = len(unique_labels)
        self._head = torch.nn.Linear(hidden, n_classes).to(device)
        self._base_model.to(device)
        optimizer = torch.optim.Adam(self._head.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self._base_model.eval()

        val_texts = None
        val_labels = None
        if x_val is not None and y_val is not None:
            val_texts = self._extract_texts(x_val)
            val_labels = np.vectorize(label2idx.get)(y_val).astype(int)

        def batch_iter(texts_batch, labels_batch):
            for i in range(0, len(texts_batch), self.batch_size):
                yield (
                    texts_batch[i : i + self.batch_size],
                    labels_batch[i : i + self.batch_size],
                )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self._head.train()
            train_loss = 0.0
            n_batches = 0

            for bt, by_raw in batch_iter(texts, y):
                by = np.vectorize(label2idx.get)(by_raw).astype(int)
                enc = self._tokenize(bt)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = self._base_model(**enc)
                    cls = out.last_hidden_state[:, 0]

                logits = self._head(cls)
                loss = loss_fn(logits, torch.tensor(by, dtype=torch.long, device=device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0.0

            if val_texts is not None:
                self._head.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for bt, by in batch_iter(val_texts, val_labels):
                        enc = self._tokenize(bt)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        out = self._base_model(**enc)
                        cls = out.last_hidden_state[:, 0]
                        logits = self._head(cls)
                        loss = loss_fn(logits, torch.tensor(by, dtype=torch.long, device=device))
                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
                log.info(
                    "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
                    epoch + 1,
                    self.epochs,
                    avg_train_loss,
                    avg_val_loss,
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= DISTILBERT_EARLY_STOP_PATIENCE:
                        log.info(
                            "Ранняя остановка: нет улучшения за %d эпох",
                            DISTILBERT_EARLY_STOP_PATIENCE,
                        )
                        break
            else:
                log.info("Epoch %d/%d: train_loss=%.4f", epoch + 1, self.epochs, avg_train_loss)

        self._device_actual = device
        self._head.eval()
        self._fitted = True
        return self

    def _ensure_fitted(self) -> None:
        """Проверяет, что модель обучена."""
        if not self._fitted or self._base_model is None or self._head is None:
            raise RuntimeError("DistilBertClassifier не обучен")

    def predict(self, x):
        """Выполняет предсказание классов."""
        self._ensure_fitted()
        texts = self._extract_texts(x)
        preds = []
        device = self._get_device()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenize(batch)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._base_model(**enc)
                cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        preds = np.array(preds)
        return self.classes_[preds]

    def predict_proba(self, x):
        """Возвращает вероятности классов."""
        self._ensure_fitted()
        import torch.nn.functional as func

        texts = self._extract_texts(x)
        device = self._get_device()
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenize(batch)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._base_model(**enc)
                cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                probs = func.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        if not all_probs:
            return np.zeros((0, len(self.classes_)), dtype=float)
        return np.vstack(all_probs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_device_actual"] = torch.device("cpu")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._device_actual = torch.device("cpu")
