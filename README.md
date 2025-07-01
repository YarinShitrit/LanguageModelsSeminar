# Dynamic-k Retrieval for RAG

**Automatic Dynamic-k Selection Strategies to Optimize Cost and Accuracy in Retrieval-Augmented Generation**

---

## 📜 Overview

This project investigates **dynamic-k** retrieval strategies in Retrieval-Augmented Generation (RAG) systems, comparing them against fixed-k baselines. We demonstrate how statistical methods (Bonferroni correction, Benjamini–Hochberg, Higher Criticism) can adaptively determine the number of documents to retrieve per query, yielding better accuracy–cost trade-offs on question-answering datasets (HotPotQA).

## 🎯 Motivation

* **Fixed-k** retrieval retrieves a constant number of documents for every query, often leading to unnecessary token usage and cost when fewer documents suffice, or insufficient context when more are needed.
* **Dynamic-k** adapts retrieval depth per question, balancing **accuracy** and **compute/token cost** in real time, crucial for production LLM applications with strict budget and latency constraints.

## 🧬 Dataset & Embeddings

* **Dataset**: HotPotQA short-answer questions (10K+ QA pairs).
* **Embeddings**: OpenAI o4-mini embeddings via LangChain (or Hugging Face equivalents).
* **Vector store**: ChromaDB used to index document embeddings and perform top-k retrieval.

## 🛠️ Methods

1. **Null Distribution Generation**

   * Sample random pairs of question and document embeddings.
   * Compute cosine similarities to build a null distribution of similarity scores.

2. **Statistical Thresholding**

   * **Bonferroni correction** (α=0.05): adjust p-values to control family-wise error rate.
   * **Benjamini–Hochberg** (α=0.05): control the false discovery rate for dynamic-k selection.
   * **Higher Criticism** (α=0.05): identify the most informative documents by detecting signal peaks above the null.

3. **Fixed-k Baseline**

   * Evaluate RAG performance for k=1…20.
   * Track total tokens (input+output), estimated cost (using GPT o4-mini pricing), and accuracy.

4. **Parallel Evaluation**

   * Use `ThreadPoolExecutor` to query the RAG agent in parallel for speed.
   * Report accuracy, token counts, cost, and accuracy-per-1K-tokens metrics.

## 📈 Results

| Method                        | Accuracy | Tokens Used | Cost (USD) | Acc per 1K Tokens |
| ----------------------------- | -------: | ----------: | ---------: | ----------------: |
| **Fixed k = 1**               |      59% |       12.7K |      0.009 |              4.64 |
| **Fixed k = 7**               |      67% |       60.4K |      0.076 |              1.11 |
| **Fixed k = 20**              |      68% |       85.2K |      0.199 |              0.80 |
| **Higher Criticism (α=0.05)** |      79% |       43.2K |      0.040 |          **1.83** |
| **Bonferroni (α=0.05)**       |      72% |       39.3K |      0.021 |              1.83 |
| **Benjamini–Hochberg**        |      73% |       39.3K |      0.021 |              1.85 |

> **Key Insight**: Dynamic-k methods achieve up to **+12 percentage points** higher accuracy and **4–5× lower cost** compared to fixed-k retrieval at k=20.

## 📊 Plots

* `plot_tokens_cost_with_accuracy(...)` → Accuracy vs. Cost curves for fixed & dynamic-k.
* `plot_null_and_real_overlay(...)` → Overlaid histogram of null vs. real Q–Doc similarities.

## ⚙️ Installation & Usage

1. **Clone** repository:

   ```bash
   git clone https://github.com/your-org/rag-dynamic-k.git
   cd rag-dynamic-k
   ```

2. **Create & activate** Python environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure** OpenAI/LangChain credentials in `.env`:

   ```ini
   OPENAI_API_KEY=your_key
   ```

4. **Run evaluation**:

   ```bash
   python evaluate.py \
     --dataset hotpotqa \
     --dynamic-methods bonferroni bh higher_criticism \
     --fixed-k 1 2 3 ... 20 \
     --output-dir results/
   ```

5. **Visualize** results:

   ```python
   from analysis import plot_tokens_cost_with_accuracy, plot_null_and_real_overlay
   plot_tokens_cost_with_accuracy(fixed_results, dynamic_results)
   plot_null_and_real_overlay(null_dist, real_sims)
   ```

## 🚀 Future Work

* **Adaptive α-selection**: optimize significance levels per query.
* **Cost-aware retrieval**: integrate real-time cost feedback.
* **Broader benchmarks**: extend evaluation to other QA datasets (e.g., NaturalQuestions, TriviaQA).
* **Production integration**: plug dynamic-k into live RAG-serving pipelines (FastAPI + React).

## 📄 License

MIT © 2025 Your Name or Organization
