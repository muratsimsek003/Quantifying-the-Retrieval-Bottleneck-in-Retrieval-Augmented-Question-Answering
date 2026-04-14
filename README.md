# Quantifying the Retrieval Bottleneck in Retrieval-Augmented Question Answering

## Description

This repository contains the complete experimental pipeline for the paper *"Quantifying the Retrieval Bottleneck in Retrieval-Augmented Question Answering"*, submitted to PeerJ Computer Science.

The study presents a controlled factorial experiment that isolates and quantifies the contribution of the retrieval stage to overall RAG (Retrieval-Augmented Generation) performance. We introduce the **Retrieval Bottleneck Diagnostic Framework (RBDF)**, an interpretable metric that determines whether the dominant source of error in a RAG pipeline is retrieval or generation.

The experiments compare retrieved-context and gold-context generation across **27 configurations**: 3 datasets × 3 generator scales × 3 retrieval strategies, with all generation parameters held constant.

## Dataset Information

Three publicly available English question answering benchmarks are used:

| Dataset | Type | Source | URL |
|---------|------|--------|-----|
| **SQuAD** | Extractive, single-hop QA | Rajpurkar et al. (2016) | [huggingface.co/datasets/rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) |
| **HotpotQA** | Multi-hop QA | Yang et al. (2018) | [huggingface.co/datasets/hotpot_qa](https://huggingface.co/datasets/hotpot_qa) |
| **TriviaQA** | Open-domain QA | Joshi et al. (2017) | [huggingface.co/datasets/trivia_qa](https://huggingface.co/datasets/trivia_qa) |

All datasets are loaded directly from the HuggingFace Datasets library at runtime. For each dataset, 500 instances are sampled from the validation split using a fixed random seed (`seed=42`). Passage pools are constructed from the sampled instances (capped at 5,000 unique passages for HotpotQA and TriviaQA). No additional cleaning, augmentation, or transformation is applied beyond standard SQuAD-style answer normalisation.

## Code Information

The entire experimental pipeline is contained in a single Jupyter Notebook:

| File | Description |
|------|-------------|
| `retrieval_bottleneck_experiment.ipynb` | End-to-end pipeline covering data loading, passage pool construction, retrieval (BM25, Sentence-BERT, Hybrid), gold-context and retrieved-context generation (Flan-T5 Small/Base/Large), evaluation metrics (EM, F1, Recall@k, RBDF), statistical tests (bootstrap CI, Wilcoxon), and figure generation |

The notebook is structured in sequential sections and is designed to be executed top-to-bottom in a single session on Google Colab with GPU runtime.

## Usage Instructions

### Option 1: Run on Google Colab (Recommended)

1. Open the notebook in Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Select **File → Open notebook → GitHub**
   - Paste the repository URL: `https://github.com/muratsimsek003/retrievalbottleneckQA`
   - Select the `.ipynb` file
2. Set the runtime to **GPU**: **Runtime → Change runtime type → T4 GPU**
3. Run all cells sequentially: **Runtime → Run all**

### Option 2: Run Locally

```bash
git clone https://github.com/muratsimsek003/retrievalbottleneckQA.git
cd retrievalbottleneckQA
pip install transformers datasets sentence-transformers faiss-cpu rank-bm25 scipy torch numpy pandas matplotlib seaborn
jupyter notebook retrieval_bottleneck_experiment.ipynb
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended; CPU execution is possible but significantly slower)

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

On Google Colab, most of these libraries are pre-installed. The notebook includes `!pip install` commands for any missing dependencies at the top.

## Methodology

The notebook executes the following steps in order:

1. **Data Sampling**: 500 instances are sampled from each dataset's validation split using `seed=42`. Passage pools are constructed from the sampled instances.

2. **Retrieval**: Three retrieval strategies are applied to each dataset:
   - **BM25** (sparse): Okapi BM25 over whitespace-tokenised passages
   - **Sentence-BERT** (dense): `all-MiniLM-L6-v2` embeddings with FAISS flat inner-product search
   - **Hybrid**: Linear combination of min-max normalised BM25 and dense scores (α = 0.5)

   Top-10 passages are retrieved per query. Retrieval results are cached per dataset–retriever pair.

3. **Generation**: For each of the 27 configurations, two conditions are evaluated:
   - **Retrieved-context**: Generator receives the top-1 retrieved passage
   - **Gold-context**: Generator receives the ground-truth supporting passage

   Three Flan-T5 scales are used: Small (77M), Base (248M), Large (783M). All use the same prompt template and decoding parameters (`num_beams=2`, `max_new_tokens=50`).

4. **Evaluation**:
   - Exact Match (EM) and token-level F1 with SQuAD-style normalisation
   - Recall@k for k ∈ {1, 3, 5, 10}
   - Bootstrap 95% confidence intervals (1,000 resamples, seed=42)
   - Wilcoxon signed-rank tests for paired comparisons
   - **RBDF score**: `RBDF = 1 − (F1_correct_retrieval / F1_gold)`

5. **Reproducibility Controls**: Fixed random seed across Python, NumPy, and PyTorch; identical prompts and decoding settings across all conditions; 5 warm-up iterations excluded from measurements.

## Computing Infrastructure

Experiments were conducted on Google Colaboratory (Colab) with GPU-accelerated runtime:
- **OS**: Ubuntu 22.04 LTS (Colab runtime)
- **CPU**: Intel Xeon
- **RAM**: ~12.7 GB system RAM
- **GPU**: NVIDIA Tesla T4 (15 GB VRAM)

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
