---
layout: default
title: "NLP & LLMs"
description: Spring 2026 @ Fudan University
theme: jekyll-theme-minimal
---

## Outline
{: .no_toc }

* TOC
{:toc}

---

## Course Overview

### Introduction

This course covers the foundations and modern frontiers of Natural Language Processing (NLP), with a heavy emphasis on **Large Language Models (LLMs)**. You will learn the modern pipeline of building effective LLMs from basic tokenization to training, fine-tuning, and deploying modern LLM architectures.

- **Prerequisites:** No formal prerequisites. For Fudan students, it’s safe to take this course in your second year.

### Basic Info

- **Course ID:** CS40008.01: NLP and LLMs
- **Course Info:** Spring 2026, Fudan University
- **Instructor:** [Baojian Zhou](https://baojian.github.io/), Email: bjzhou AT fudan.edu.cn
- **TAs:**
  - Binbin Huang
  - Runze Wang
- **Time:** 03/05/2026–06/18/2026, Thu., 1:30 pm – 4:10 pm, <a href="assets/Fudan-2026-spring-calendar.pdf" target="_blank" rel="noopener">Fudan Calendar</a>
- **Location:** HGX104
- **Office Hours:** 10:00 am – 1:00 pm, Wed & C611



---

## Assignments and Course Project

All standard homework assignments are completed by **Week 12**. The final month (Weeks 13–16) is dedicated exclusively to the Course Project. Please submit your homework at [https://elearning.fudan.edu.cn/](https://elearning.fudan.edu.cn/)

### Assignment 1. Foundations of Text
- **Due:** Week 5 (04/02/2026 23:59 pm)
- **Deliverable:** Please check out it at [elearning](https://elearning.fudan.edu.cn/).


### Course Project
- **Timeline:** Weeks 10–16 (Dedicated time)
- **Teams:** 3-5 Students (3 members are strongly recommended).
- **Deliverables:**
  - **Week 7:** Make your team  members ready
  - **Week 9:** 1-page Project Proposal (Problem, Dataset, Baselines).
  - **Week 12:** Status Check / Preliminary Results (Presentation).
  - **Week 17:** Final Report / Code submission
- **[Course project details](https://baojian.github.io/llm-26/slides/final-project/index.html)**

---

## Coursework

- **Participation (5%)**
- **Exercise (15%)**
- **Assignments (40-45%)**
- **Final Project (35-40%)**


## Resources and References

- **Textbook & Readings**
  - [Speech and Language Processing (Jurafsky & Martin, 3rd Ed. Draft)](https://web.stanford.edu/~jurafsky/slp3/)
  - [Natural Language Processing: Neural Networks and Large Language Models](https://niutrans.github.io/NLPBook/)
- **Computing & Communication**
  - **University Cluster:** [https://cfff.fudan.edu.cn/home](https://cfff.fudan.edu.cn/home)
  - **We will use Qizhi**: [http://qz.cfff.fudan.edu.cn/](http://qz.cfff.fudan.edu.cn/)
  - **Course Repo:** [baojian/llm-26](https://github.com/baojian/llm-26)
  - **Academic Integrity:** Please check [our policies](https://baojian.github.io/llm-26/assets/Fudan-academic-integrity.pdf).
- **AI Policy**
Use of AI coding assistants **is permitted**. However, you should explicitly attribute the use of AI in your assignment.

---

## GPU Resources

- **Please register CFFF an account at [Fudan CFFF](http://cfff.fudan.edu.cn/home).**

---

## Weekly Schedule

### Week 1 Introduction to LLMs

> In our first lecture, we introduce text preprocessing, including tokenization (BPE/WordPiece), and vocab design.
> 
> - Thu., 1:30 pm – 4:10 pm, 03/05/2026
> - **Slides:** <a href="slides/lecture-01-slides/index.html" target="_blank" rel="noopener">Lecture 01 slides</a>
> - **Readings:**
>   - <a href="papers/lecture-01-readings-0-JM_Book_Chapter_2.pdf" target="_blank" rel="noopener">JM Book Chapter 2</a>
>   - <a href="papers/lecture-01-readings-1-Advances_in_NLP-2015.pdf" target="_blank" rel="noopener">Advances in NLP</a>
>   - <a href="papers/lecture-01-readings-2-Human_Language_Understanding_and_Reasoning-2022.pdf" target="_blank" rel="noopener">Human Language Understanding and Reasoning</a>
>   - <a href="papers/lecture-01-readings-3-Scaling_Laws_with_Vocabulary-2024.pdf" target="_blank" rel="noopener">Scaling Laws with Vocabulary (2407.13623v1)</a>
>   - <a href="papers/lecture-01-readings-4-Getting_the_most_out_of_your_tokenizer_for_pre-training-2024.pdf" target="_blank" rel="noopener">Getting the most out of your tokenizer for pre-training</a>
> - **Excercise: <a href="https://github.com/baojian/llm-26/blob/main/lecture-01-tokenization/lecture-01-exercise-tokenization.ipynb" target="_blank" rel="noopener">lecture-01-exercise-tokenization.ipynb</a>**

**Release Assignment 1**

### Week 2 N-Gram Language Models

> In this lecture, we introduce the concept of MLE, Smoothing, Perplexity, and Language Modeling basics.
> - **Slides:** <a href="slides/lecture-02-slides/index.html" target="_blank" rel="noopener">Lecture 02 slides</a>
> - **Readings:**
>   - <a href="papers/lecture-02-readings-0-JM_Book_Chapter_3.pdf" target="_blank" rel="noopener">JM Book Chapter 3</a>
>   - <a href="papers/lecture-02-readings-1-chen-goodman-99-v2.pdf" target="_blank" rel="noopener">N-gram smoothing</a>
>   - <a href="papers/lecture-02-readings-2-Google-Translate-06.pdf" target="_blank" rel="noopener">First version of Google Translate (2006)</a>
>   - <a href="papers/lecture-02-readings-3-Joshua-Goodman.pdf" target="_blank" rel="noopener">Progress of language modeling (2009)</a>
>   - <a href="papers/lecture-02-readings-4-infinity-gram.pdf" target="_blank" rel="noopener">Application of Infinity-gram model</a>
> - **Excercise: <a href="https://github.com/baojian/llm-26/blob/main/lecture-02-ngram-lm/lecture-02-ngram-lms.ipynb" target="_blank" rel="noopener">lecture-02-ngram-lms.ipynb</a>**

### Week 3 Word Embeddings

> In this lecture, we introduce text classification, Word2Vec, Distributional Hypothesis, and Intrinsic/Extrinsic evaluations.
> - **Slides:** <a href="slides/lecture-03-slides/index.html" target="_blank" rel="noopener">Lecture 03 slides</a>
> - **Readings:**
>   - <a href="papers/lecture-03-readings-0-JM_Book_Chapter_4_5.pdf" target="_blank" rel="noopener">JM Book Chapter 4-5</a>
>   - <a href="papers/lecture-03-readings-1-word2vec1.pdf" target="_blank" rel="noopener">Word2vec (2013)</a>
>   - <a href="papers/lecture-03-readings-2-word2vec2.pdf" target="_blank" rel="noopener">CBOW (2013)</a>
>   - <a href="papers/lecture-03-readings-3-qwen-embeddings.pdf" target="_blank" rel="noopener">Qwen Embeddings (2025)</a>
>   - <a href="papers/lecture-03-readings-4-gemini-embeddings.pdf" target="_blank" rel="noopener">Gemini Embeddings (2025)</a>
> - **Excercise: <a href="https://github.com/baojian/llm-26/blob/main/lecture-03-embeddings/lecture-03-embeddings.ipynb" target="_blank" rel="noopener">lecture-03-embeddings.ipynb</a>**

### Week 4 Neural LMs

> In this lecture, we introduce neural networks and how to build NN models for sequence learning problems.
> We will discuss some classic models like LSTM and how the encoder-decoder style models developed and why the attention
> is a effective component adding to encoder-decoder model.
> - **Slides:** <a href="https://github.com/baojian/llm-26/blob/main/slides/lecture-04-slides/lecture-04-slides.pdf" target="_blank" rel="noopener">Lecture 04 slides</a>
> - **Readings:**
>   - <a href="papers/lecture-04-readings-0-JM_Book_Chapter_6_13.pdf" target="_blank" rel="noopener">JM Book Chapter 6,13</a>
>   - <a href="papers/lecture-04-readings-1-NPLM.pdf" target="_blank" rel="noopener">NPLM (2003)</a>
>   - <a href="papers/lecture-04-readings-2-Revisiting_Simple_NPLM.pdf" target="_blank" rel="noopener">Revisiting NPLM (2021)</a>
>   - <a href="papers/lecture-04-readings-3-LSTM.pdf" target="_blank" rel="noopener">LSTM (1997)</a>
>   - <a href="papers/lecture-04-readings-4-Attention-RNN.pdf" target="_blank" rel="noopener">Attention RNN (2015)</a>
> - **Excercise: <a href="https://github.com/baojian/llm-26/blob/main/lecture-04-neural-lms/lecture-04-neural-lms.ipynb" target="_blank" rel="noopener">lecture-04-neural-lms.ipynb</a>**


### Week 5 Attention Mechanisms and Transformer

> In this lecture, we introduce the Transformer architecture.
> - **Slides:** <a href="https://github.com/baojian/llm-26/blob/main/slides/lecture-05-slides/index.html" target="_blank" rel="noopener">Lecture 05 slides</a>
> - **Readings:**
>   - <a href="papers/lecture-05-readings-0-JM_Book_Chapter_7_8.pdf" target="_blank" rel="noopener">JM Book Chapter 7,8</a>
>   - <a href="papers/lecture-05-readings-2-Decomposable-Attention-2017.pdf" target="_blank" rel="noopener">Decomposable Attention (2017)</a>
>   - <a href="papers/lecture-05-readings-1-Transformer-paper.pdf" target="_blank" rel="noopener">Transformer (2017)</a>
>   - <a href="papers/lecture-05-readings-3-Transformer-in-equation-2019.pdf" target="_blank" rel="noopener">Transformer in Equation (2019)</a>
> - **Excercise: <a href="https://github.com/baojian/llm-26/blob/main/lecture-05-transformers/lecture-05-transformers.ipynb" target="_blank" rel="noopener">lecture-05-transformers.ipynb</a>**


### Week 7 LLM Pretraining

Causal LM vs MLM, Chinchilla Scaling Laws, Data Mixtures

### Week 8 Fine-tuning & PEFT

SFT, LoRA/QLoRA, Adapters, Instruction Tuning

**Project Proposal Due**

### Week 10 Evaluation

Benchmarks (MMLU/GSM8K), Contamination, LLM-as-a-judge

### Week 11 Prompt Engineering

In-context learning, Chain-of-Thought, Prompt sensitivity 

### Week 12 Course Project Presentation

### Week 13 RAG Systems

Dense Retrieval, Vector DBs, Reranking, Grounding

### Week 14 Alignment & Safety

RLHF (PPO/DPO), Safety barriers, Red-teaming

### Week 15 Efficiency & Systems

KV Caching, Quantization (Int8/FP4), Latency/Throughput

### Week 16 Agents and Frontiers

Multimodal LLMs, Diffusion LMs, Future Directions


