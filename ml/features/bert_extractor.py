import sys
import numpy as np
import pandas as pd
import torch
from typing import List, Optional
from transformers import AutoModel
from tqdm import tqdm
from sklearn.decomposition import PCA

class CodeBertExtractor:
    def __init__(
        self, 
        model_name: str = "jinaai/jina-embeddings-v2-base-code", 
        batch_size: int = 8, 
        max_length: int = 1024,
        device: str = None,
        use_pca: bool = False,    
        n_components: int = 32    
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.pca = None
        self.is_fitted = False
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Jina Embeddings initialized on {self.device}")
        if self.use_pca:
            print(f"PCA enabled: reducing to {self.n_components} components")
        
        # trust_remote_code=True обязателен для Jina
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def extract(self, code_samples: List[str]) -> pd.DataFrame:
        embeddings = []
        # print(f"Генерация эмбеддингов ({len(code_samples)} примеров)...")
        
        # 1. Получаем сырые векторы (768)
        for i in tqdm(range(0, len(code_samples), self.batch_size), desc="Embedding", disable=True):
            batch_code = code_samples[i : i + self.batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch_code, max_length=self.max_length)
            embeddings.append(batch_embeddings)
                
        if embeddings:
            full_embeddings = np.vstack(embeddings)
        else:
            full_embeddings = np.zeros((0, 768))
            
        # 2. Применяем PCA если нужно
        if self.use_pca:
            if not self.is_fitted:
                # Обучаем PCA на первом батче (Train)
                # print("  Training PCA...")
                self.pca = PCA(n_components=self.n_components)
                full_embeddings = self.pca.fit_transform(full_embeddings)
                self.is_fitted = True
                # print(f"  PCA explained variance: {sum(self.pca.explained_variance_ratio_):.2f}")
            else:
                # Применяем обученный PCA (Val/Test)
                full_embeddings = self.pca.transform(full_embeddings)
        
        # 3. Формируем DataFrame
        prefix = "pca_bert" if self.use_pca else "jina"
        cols = [f"{prefix}_{i}" for i in range(full_embeddings.shape[1])]
        
        return pd.DataFrame(full_embeddings, columns=cols)
