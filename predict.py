# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input
import os
import time
import torch
import subprocess
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel


MODEL_ID = "Qwen3-Embedding-8B"
MODEL_URL = "https://weights.replicate.delivery/default/Qwen/Qwen3-Embedding-8B/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights
        if not os.path.exists(MODEL_ID):
            download_weights(MODEL_URL, MODEL_ID)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set tokenizer padding side to left as recommended
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        print(f"Model loaded successfully")

    def predict(
        self,
        text: Union[str, List[str]] = Input(
            description="Text or list of texts to embed"
        ),
        instruction: str = Input(
            description="Optional instruction for task-specific embeddings",
            default=""
        ),
        embedding_dim: int = Input(
            description="Output embedding dimension",
            ge=32,
            le=4096,
            default=1024
        ),
        normalize: bool = Input(
            description="Whether to normalize embeddings",
            default=True
        ),
        batch_size: int = Input(
            description="Batch size for processing multiple texts",
            ge=1,
            le=32,
            default=8
        )
    ) -> dict:
        """Run embedding generation on the model"""
        
        # Ensure text is a list
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        # Prepare texts with instruction if provided
        if instruction:
            texts = [f"{instruction}\n{t}" for t in texts]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=32768,
                return_tensors="pt"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use mean pooling of last hidden state
                last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Apply attention mask for mean pooling
                masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
                summed_embeddings = masked_embeddings.sum(dim=1)
                sequence_lengths = attention_mask.sum(dim=1)
                
                batch_embeddings = summed_embeddings / sequence_lengths.unsqueeze(-1)
                
                # Adjust embedding dimension if needed
                if embedding_dim != batch_embeddings.shape[-1]:
                    batch_embeddings = batch_embeddings[:, :embedding_dim]
                
                # Normalize if requested
                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        final_embeddings = np.concatenate(embeddings, axis=0)
        
        # Prepare output
        result = {
            "embeddings": final_embeddings.tolist(),
            "shape": list(final_embeddings.shape),
            "model": MODEL_ID,
            "embedding_dim": embedding_dim,
            "normalized": normalize,
            "instruction": instruction if instruction else None,
            "num_texts": len(texts)
        }
        
        return result
