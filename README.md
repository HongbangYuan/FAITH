# FAITH

## Description

This repository contains the official code and data for our paper titled
"[Whispers that Shake Foundations: Analyzing and Mitigating 
False Premise Hallucinations in Large Language Models](https://arxiv.org/pdf/2402.19103)", accepted by EMNLP 2024. 

## Core Concept: How to observe and manipulate the hidden state in LLMs?
 
A big part of FAITH is about looking under the hood of language models. To do this, we borrowed inspiration from the excellent  [baukit](https://github.com/davidbau/baukit/tree/main)
 toolkit, which provides utilities for probing and editing neural networks.

We streamlined this into a single file: `./utils/nethook.py`
.
Here, you’ll find:

1. Hooks for observing and manipulating hidden states.

2. A new `edit_input` feature we added for directly manipulate the inputs for any layer inside a model.

This way, you don’t need to jump across multiple files—everything is wrapped in one place, making the workflow smooth and easy to adapt.