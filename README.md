# ATMM-SAGA: Score-Aware Gated Attention with Alternating Training for Robust Speaker Verification

Official implementation of the paper:  
📄 **[ATMM-SAGA: Score-Aware Gated Attention with Alternating Training for Robust Speaker Verification](https://arxiv.org/abs/2408.00001)**  
Presented at **Interspeech 2025**.

## 🧠 Overview

This repository provides the code for a spoofing-robust automatic speaker verification (SASV) system that combines:
- **AASIST**: A spoofing countermeasure model
- **ECAPA-TDNN**: A speaker verification model
- **SAGA**: A simple multiplicative attention mechanism gating ASV embeddings using CM scores
- **ATMM**: Alternating Training for Multi-Module, which switches optimization between ASV and CM tasks to prevent overfitting and encourage balanced learning

Our approach significantly improves SASV performance on the ASVspoof 2019 LA dataset, achieving:
- **2.31% SASV-EER / 0.0603 min a-DCF** on the development set
- **2.18% SASV-EER / 0.0480 min a-DCF** on the evaluation set

## 📂 Repository Structure

