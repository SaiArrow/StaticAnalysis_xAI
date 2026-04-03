### Relevance Training using Human Annotations

**Table 1: Interpretability metrics for InputxGradient (IxG) and Integrated Gradients (IG) using human annotations**

| Model | XAI | App | JBMC T@1/3/5 | JBMC IFA | JBMC R@T | Jayhorn T@1/3/5 | Jayhorn IFA | Jayhorn R@T |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek** | IxG | KL | 71/98/98 | 0.4 | 55.8 | 62/92/97 | 0.7 | 51.8 |
| | | EGL | 83/98/98 | 0.4 | 56.8 | 74/95/97 | 0.5 | 56.4 |
| | IG | KL | 80/88/95 | 1.2 | 45.0 | 41/82/92 | 1.3 | 40.5 |
| | | EGL | 60/88/95 | 1.5 | 43.7 | 80/97/100 | 0.4 | 49.3 |
| **Stable-Code** | IxG | KL | 78/88/93 | 1.1 | 45.2 | 51/92/95 | 0.7 | 51.2 |
| | | EGL | 61/90/97 | 1.3 | 49.2 | 64/87/100 | 0.6 | 54.6 |
| | IG | KL | 25/76/92 | 1.9 | 34.0 | 33/82/97 | 1.2 | 41.9 |
| | | EGL | 30/88/97 | 1.1 | 39.3 | 31/87/97 | 1.2 | 42.0 |

In Section V (RQ2), we utilized automated program slicing (via WALA and LLVM-Slicer) to generate surrogate rationales. While automated slicing ensures scalability for large training datasets while covering a high percentage of the true relevant lines, it inherently introduces a degree of noise and over-approximation by including structurally dependent but semantically irrelevant lines. To assess our approach's sensitivity to these slicer inaccuracies, Table 1 explores the upper bound of relevance-guided training by replacing the automated slices with gold-standard human annotations.

While Table 1 shows human annotations yield near-perfect explanations (e.g., DeepSeek achieving an 83% Top-1 hit rate and 0.4 IFA on JBMC), manual labeling is prohibitively expensive. Crucially, comparing these ideal conditions to our slice-trained results reveals that automated slicing achieves highly comparable performance on broader retrieval metrics without the severe bottleneck of manual curation. For example, on the JBMC dataset, slicer-derived rationales guide DeepSeek to a 95% Top-5 hit rate, which is highly comparable to the 98% achieved with human annotations. Similarly, under EGL training, Stable-Code achieves an IFA of 1.4 using automated slices, nearly matching the 1.3 IFA achieved with human labels. This illustrates the necessary and favorable trade-off inherent in automated slicing: despite structural over-approximation, it is sufficiently robust to deliver comparable, scalable performance. Scalable automated tools provide a highly practical pathway to drastically reduce developer triage effort without the prohibitive expense of human curation.
