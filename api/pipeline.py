import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ML libraries
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.decomposition import PCA

app = FastAPI()

# Load resources once (cached across invocations)
# 1. BERT tokenizer (used for token IDs)
BERT_TOKENIZER = Tokenizer.from_pretrained("bert-base-uncased")

# 2. MiniLM model for embeddings
MINILM_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3. GPT-2 tokenizer + model for generation (small, 124M)
GPT2_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
GPT2_MODEL = AutoModelForCausalLM.from_pretrained("gpt2")

# Ensure models are on CPU (Vercel free tier) and in eval mode
MINILM_MODEL.eval()
GPT2_MODEL.eval()

class PipelineRequest(BaseModel):
    text: str
    model: str  # "bert", "minilm", "gpt2"

class PipelineResponse(BaseModel):
    pipeline: List[str]
    tokens: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
    embedding: Optional[List[float]] = None
    embedding_2d: Optional[List[float]] = None
    generation: Optional[str] = None
    info: Optional[Dict[str, Any]] = None

def tokenise(text: str):
    enc = BERT_TOKENIZER.encode(text)
    return enc.tokens, enc.ids

def embed(text: str):
    vec = MINILM_MODEL.encode(text, convert_to_numpy=True)
    # PCA to 2‑D (single vector -> we just take first two dims)
    # For consistency we still run PCA on the vector reshaped as (1, dim)
    pca = PCA(n_components=2)
    vec2d = pca.fit_transform(vec.reshape(1, -1)).flatten().tolist()
    return vec.tolist(), vec2d

def generate(text: str):
    input_ids = GPT2_TOKENIZER.encode(text, return_tensors="pt")
    # Limit generation length to keep within time limits
    output_ids = GPT2_MODEL.generate(input_ids, max_new_tokens=30, do_sample=True, temperature=0.8)
    gen_text = GPT2_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
    return gen_text

@app.post("/api/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    steps = []
    response = PipelineResponse(pipeline=[])

    # Tokenisation step is common for all models
    tokens, ids = tokenise(req.text)
    response.tokens = tokens
    response.token_ids = ids
    steps.append("Tokenisation")

    if req.model == "bert":
        response.pipeline = steps
        response.info = {"note": "Only tokenisation performed for BERT tokenizer mode."}
        return response

    if req.model == "minilm":
        emb, emb2d = embed(req.text)
        response.embedding = emb
        response.embedding_2d = emb2d
        steps.append("Embedding (MiniLM)")
        response.pipeline = steps
        return response

    if req.model == "gpt2":
        gen = generate(req.text)
        response.generation = gen
        steps.append("Generation (GPT‑2)")
        response.pipeline = steps
        return response

    # Fallback – unknown model
    response.info = {"error": f"Unsupported model '{req.model}'."}
    response.pipeline = steps
    return response
