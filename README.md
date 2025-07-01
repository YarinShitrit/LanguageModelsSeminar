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
