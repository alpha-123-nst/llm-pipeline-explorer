# LLM Pipeline Explorer

A **single‑page Vercel web app** that mimics the functionality of the public LLM‑Pipeline Explorer demo. It lets you:

1. **Select a model** (BERT tokenizer, MiniLM embedding, GPT‑2 text generation).
2. **Enter a prompt**.
3. **View tokenisation** (tokens & IDs).
4. **See the embedding** visualised in a 2‑D PCA plot.
5. **Get a generation** from a small language model.
6. **Observe the pipeline steps** as a horizontal flow diagram.

All heavy work runs in a **Python serverless function** (FastAPI) on Vercel, cached across requests, so the app stays within the free tier.

## Project structure
```
llm-pipeline-explorer/
├─ api/
│   └─ pipeline.py          # FastAPI endpoint handling all steps
├─ public/
│   └─ index.html           # UI (Tailwind CSS + Plotly.js)
├─ requirements.txt         # Python dependencies
├─ vercel.json              # Vercel config (static + python function)
└─ README.md                # This file
```

## Deploy – no terminal needed
1. **Compress** the whole `llm-pipeline-explorer` folder into a ZIP file.
2. In the Vercel dashboard click **New Project → Import → Upload a ZIP** and select the ZIP.
3. Choose **Framework = Other**, keep the root as `./`, and click **Deploy**.
4. After the build finishes you’ll get a URL like `https://llm‑pipeline‑explorer‑xxxxx.vercel.app`.

Open that URL, type a prompt, pick a model, and explore the pipeline!
