# Quantifying the Retrieval Bottleneck in Retrieval-Augmented Question Answering

## Description

This repository contains the complete experimental pipeline for the paper *"Quantifying the Retrieval Bottleneck in Retrieval-Augmented Question Answering"*, submitted to PeerJ Computer Science. The study presents a controlled factorial experiment that isolates and quantifies the contribution of the retrieval stage to overall RAG (Retrieval-Augmented Generation) performance. We introduce the **Retrieval Bottleneck Diagnostic Framework (RBDF)**, an interpretable metric that determines whether the dominant source of error in a RAG pipeline is retrieval or generation.

The experiments compare retrieved-context and gold-context generation across **27 configurations**: 3 datasets × 3 generator scales × 3 retrieval strategies, with all generation parameters held constant.

## Dataset Information

Three publicly available English question answering benchmarks are used:

| Dataset | Type | Source | URL |
|---------|------|--------|-----|
| **SQuAD** | Extractive, single-hop QA | Rajpurkar et al. (2016) | [huggingface.co/datasets/rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) |
| **HotpotQA** | Multi-hop QA | Yang et al. (2018) | [huggingface.co/datasets/hotpot_qa](https://huggingface.co/datasets/hotpot_qa) |
| **TriviaQA** | Open-domain QA | Joshi et al. (2017) | [huggingface.co/datasets/trivia_qa](https://huggingface.co/datasets/trivia_qa) |

For each dataset, 500 instances are sampled from the validation split using a fixed random seed (`seed=42`). Passage pools are constructed from the sampled instances (capped at 5,000 unique passages for HotpotQA and TriviaQA).

## Code Information

| File / Directory | Description |
|------------------|-------------|
| `main_experiment.py` | Main experimental pipeline: data loading, retrieval, generation, and evaluation across all 27 configurations |
| `retrieval.py` | Implementation of BM25, Sentence-BERT, and hybrid retrieval strategies |
| `generation.py` | Gold-context and retrieved-context generation using Flan-T5 models |
| `evaluation.py` | Computation of Exact Match, token-level F1, Recall@k, bootstrap confidence intervals, Wilcoxon signed-rank tests, and RBDF scores |
| `utils.py` | Answer normalisation, data sampling, and helper functions |
| `requirements.txt` | Python dependencies with version numbers |
| `results/` | Output directory for experimental results (CSV files and figures) |

> **Note:** File names above reflect the intended structure. Please refer to the actual repository contents for the exact filenames.

## Usage Instructions

### 1. Clone the repository

```bash
git clone https://github.com/muratsimsek003/retrievalbottleneckQA.git
cd retrievalbottleneckQA
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full experiment

```bash
python main_experiment.py
```

This script will:
- Download and sample 500 instances from each dataset via the HuggingFace Datasets library
- Build passage pools and compute retrieval results (BM25, Sentence-BERT, Hybrid) for each dataset
- Run gold-context and retrieved-context generation for all 27 configurations using Flan-T5 Small, Base, and Large
- Compute evaluation metrics (EM, F1, Recall@k, RBDF) with bootstrap confidence intervals and statistical tests
- Save results to the `results/` directory

### 4. Reproduce specific components

To run only retrieval or only generation for a specific configuration, refer to the individual module files (`retrieval.py`, `generation.py`, `evaluation.py`) which can be executed independently.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended; CPU execution is possible but slow)

### Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `transformers` | ≥ 4.36 | Flan-T5 model loading and inference |
| `datasets` | ≥ 2.14 | Loading SQuAD, HotpotQA, TriviaQA from HuggingFace |
| `sentence-transformers` | ≥ 2.2 | Dense retrieval with all-MiniLM-L6-v2 |
| `faiss-cpu` | ≥ 1.7.4 | FAISS index for dense retrieval |
| `rank-bm25` | ≥ 0.2.2 | BM25 sparse retrieval |
| `scipy` | ≥ 1.11 | Wilcoxon signed-rank tests |
| `torch` | ≥ 2.0 | PyTorch backend for model inference |
| `numpy` | ≥ 1.24 | Numerical computations |
| `pandas` | ≥ 2.0 | Results aggregation and export |
| `matplotlib` | ≥ 3.7 | Figure generation |
| `seaborn` | ≥ 0.12 | Heatmap visualisation |

Install all dependencies at once:

```bash
pip install transformers datasets sentence-transformers faiss-cpu rank-bm25 scipy torch numpy pandas matplotlib seaborn
```

## Methodology

The experimental pipeline follows these steps:

1. **Data Sampling**: 500 instances are sampled from each dataset's validation split using `seed=42`. Passage pools are constructed from the sampled instances.

2. **Retrieval**: Three retrieval strategies are applied to each dataset:
   - **BM25** (sparse): Okapi BM25 over whitespace-tokenised passages
   - **Sentence-BERT** (dense): `all-MiniLM-L6-v2` embeddings with FAISS flat inner-product search
   - **Hybrid**: Linear combination of min-max normalised BM25 and dense scores (α = 0.5)
   
   Top-10 passages are retrieved per query. Retrieval results are cached per dataset–retriever pair.

3. **Generation**: For each configuration, two conditions are evaluated:
   - **Retrieved-context**: Generator receives the top-1 retrieved passage
   - **Gold-context**: Generator receives the ground-truth supporting passage
   
   Three Flan-T5 scales are used: Small (77M), Base (248M), Large (783M). All use the same prompt template and decoding parameters (`num_beams=2`, `max_new_tokens=50`).

4. **Evaluation**:
   - Exact Match (EM) and token-level F1 with SQuAD-style normalisation
   - Recall@k for k ∈ {1, 3, 5, 10}
   - Bootstrap 95% confidence intervals (1,000 resamples)
   - Wilcoxon signed-rank tests for paired comparisons
   - **RBDF score**: `RBDF = 1 − (F1_correct_retrieval / F1_gold)`

5. **Reproducibility Controls**: Fixed random seed across Python, NumPy, and PyTorch; identical prompts and decoding settings across all conditions; 5 warm-up iterations excluded from measurements.

## Computing Infrastructure

Experiments were conducted on:
- **OS**: Ubuntu 22.04 LTS
- **CPU**: Intel Core i7-12700K (12 cores, 3.6 GHz)
- **RAM**: 64 GB DDR5
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM)

Total wall-clock time for all 27 configurations: approximately 8 hours.

## Citations

If you use this code or the RBDF framework in your research, please cite:

```bibtex
@article{Simsek2025retrieval,
  title   = {Quantifying the Retrieval Bottleneck in Retrieval-Augmented Question Answering},
  author  = {Simsek, Murat},
  journal = {PeerJ Computer Science},
  year    = {2025},
  note    = {Under review}
}
```

### Dataset References

- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP*.
- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP*.
- Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. *ACL*.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contribution Guidelines

Contributions are welcome. Please open an issue first to discuss proposed changes. For pull requests:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with clear messages
4. Push to your fork and submit a pull request

All code contributions should maintain the existing reproducibility controls (fixed seeds, consistent parameters).
