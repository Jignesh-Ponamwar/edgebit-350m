#!/usr/bin/env python3
"""Retrieval-Augmented Generation (RAG) demo.

Demonstrates knowledge-grounded Q&A by retrieving relevant context
from a local document store and using EdgeBit to generate answers.

Uses simple TF-IDF retrieval (no external vector DB required) to
keep the demo self-contained and edge-deployable.

Usage:
    python demos/rag_demo.py --preset tiny
    python demos/rag_demo.py --checkpoint /path/to/ckpt --docs /path/to/docs/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM

RAG_PROMPT = """### Context:
{context}

### Question:
{question}

### Answer:
"""

SAMPLE_DOCS = [
    {
        "title": "EdgeBit Architecture",
        "text": "EdgeBit-350M uses ternary weights where each weight can only be -1, 0, or +1. "
                "This enables 16x compression compared to float32 storage. The model uses "
                "Grouped Query Attention with 16 query heads and 4 key-value heads, reducing "
                "KV cache memory by 4x. The feed-forward network uses SwiGLU activation with "
                "a gate, up, and down projection.",
    },
    {
        "title": "Quantization Curriculum",
        "text": "Training uses a progressive quantization curriculum with four phases: "
                "BF16 warmup for 10% of steps, INT8 adaptation for 20%, INT4 adaptation for 25%, "
                "and ternary training for the final 45%. Each transition causes a temporary loss "
                "spike that recovers within a few hundred steps.",
    },
    {
        "title": "Deployment",
        "text": "EdgeBit-350M is designed for edge deployment on devices with limited RAM. "
                "The packed model fits in approximately 156MB of memory, making it suitable "
                "for Raspberry Pi 5 (4GB RAM), commodity laptops, and cloud CPU instances. "
                "Expected decode throughput is 30-50 tokens per second on x86 CPUs.",
    },
    {
        "title": "Training Infrastructure",
        "text": "The model is trained using AdamW optimizer with learning rate 3e-4 and "
                "cosine annealing schedule. Gradient clipping at 1.0 is essential for ternary "
                "training stability. The full 350M model can be trained in 50-60 A100-hours "
                "across three stages: tiny validation, 125M proof, and 350M final.",
    },
    {
        "title": "Weight Packing",
        "text": "Ternary weights are packed into 2-bit representation with 4 values per byte. "
                "The encoding uses 00 for zero, 01 for +1, and 10 for -1. Each group of 128 "
                "weights shares a float16 scale factor. This achieves approximately 7.4x "
                "compression compared to float16 storage.",
    },
]

SAMPLE_QUESTIONS = [
    "How much memory does EdgeBit need?",
    "What are the training phases?",
    "How are weights compressed?",
    "What optimizer is used for training?",
    "What is the decode speed on CPU?",
]


class SimpleRetriever:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self):
        doc_count = len(self.documents)
        doc_freq: Counter[str] = Counter()

        for doc in self.documents:
            tokens = set(self._tokenize(doc["text"]))
            for token in tokens:
                doc_freq[token] += 1

        self.idf = {
            token: math.log(doc_count / (freq + 1))
            for token, freq in doc_freq.items()
        }

        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            tf = Counter(tokens)
            total = len(tokens)
            vec = {
                token: (count / total) * self.idf.get(token, 0)
                for token, count in tf.items()
            }
            self.doc_vectors.append(vec)

    def _query_vector(self, query: str) -> dict[str, float]:
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        total = len(tokens)
        return {
            token: (count / total) * self.idf.get(token, 0)
            for token, count in tf.items()
        }

    def _cosine_sim(self, a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a.keys()) & set(b.keys())
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 2) -> list[tuple[dict, float]]:
        q_vec = self._query_vector(query)
        scores = [
            (doc, self._cosine_sim(q_vec, d_vec))
            for doc, d_vec in zip(self.documents, self.doc_vectors)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def load_model(args):
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        config = EdgeBitConfig(**cfg_dict.get("model", cfg_dict))
    else:
        preset_fn = getattr(EdgeBitConfig, args.preset, None)
        config = preset_fn() if preset_fn else EdgeBitConfig.tiny()

    config.vocab_size = 151936
    model = EdgeBitForCausalLM(config)

    if args.checkpoint:
        ckpt_file = args.checkpoint
        if os.path.isdir(ckpt_file):
            ckpt_file = os.path.join(ckpt_file, "training_state.pt")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))

    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, config, tokenizer


def answer_question(model, tokenizer, retriever, question, max_new_tokens=64, temperature=0.3):
    results = retriever.retrieve(question, top_k=2)
    context = "\n\n".join(f"[{doc['title']}]: {doc['text']}" for doc, _ in results)
    prompt = RAG_PROMPT.format(context=context, question=question)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    elapsed = time.perf_counter() - t0

    generated = output[0, input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    n_tokens = len(generated)

    return answer, elapsed, n_tokens, [(doc["title"], score) for doc, score in results]


def main():
    parser = argparse.ArgumentParser(description="EdgeBit RAG Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--docs", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model, config, tokenizer = load_model(args)
    params = model.count_parameters()
    print(f"Model: {params['total']:,} params ({config.quant_mode} mode)")

    if args.docs:
        documents = []
        for fname in os.listdir(args.docs):
            if fname.endswith((".txt", ".json")):
                with open(os.path.join(args.docs, fname)) as f:
                    text = f.read()
                documents.append({"title": fname, "text": text})
    else:
        documents = SAMPLE_DOCS

    retriever = SimpleRetriever(documents)
    print(f"Indexed {len(documents)} documents\n")

    if args.interactive:
        print("Ask questions (type 'quit' to exit):\n")
        while True:
            try:
                question = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in ("quit", "exit"):
                break

            answer, elapsed, n_tok, sources = answer_question(
                model, tokenizer, retriever, question, args.max_tokens,
            )
            tps = n_tok / elapsed if elapsed > 0 else 0
            print(f"A: {answer}")
            print(f"  Sources: {', '.join(f'{t} ({s:.2f})' for t, s in sources)}")
            print(f"  [{n_tok} tokens, {elapsed:.2f}s, {tps:.1f} tok/s]\n")
    else:
        print("Sample Q&A:\n")
        for question in SAMPLE_QUESTIONS:
            answer, elapsed, n_tok, sources = answer_question(
                model, tokenizer, retriever, question, args.max_tokens,
            )
            tps = n_tok / elapsed if elapsed > 0 else 0
            print(f"Q: {question}")
            print(f"A: {answer}")
            print(f"  Sources: {', '.join(f'{t} ({s:.2f})' for t, s in sources)}")
            print(f"  [{n_tok} tokens, {elapsed:.2f}s, {tps:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
