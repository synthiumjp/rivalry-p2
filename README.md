# Crewther Sampler Programme — Paper 2

Pre-registered computational study of hallucination in transformer language
models. The programme tests whether hallucination is a dynamical competition
resolved across the generation trajectory, or a property set at prompt
encoding. Two contributions: a convergent-negative result against the
competition account across six independent signatures, and an encoding-locus
result showing hallucination propensity is readable from the prompt encoding
before generation.

Author: JP Cacioli — https://synthiumjp.github.io/

Pre-registration: OSF <registration-id>  (confirm before making public)

## Attribution
Builds on the H-Neurons method: Cheng Gao, Huimin Chen, Chaojun Xiao,
Zhiyi Chen, Zhiyuan Liu, Maosong Sun. *H-Neurons: On the Existence, Impact,
and Origin of Hallucination-Associated Neurons in LLMs.* Upstream repository:
https://github.com/thunlp/H-Neurons . All pipeline code here is an independent
reimplementation (HuggingFace + MPS).

## Status
Benchmark constructed and locked (Cat1 1000, Cat2 500, Cat3 250). Confirmatory
five-model collection not yet started. See the benchmark construction and
deviation documents in this repository.

## Licence
All rights reserved. See LICENSE.
