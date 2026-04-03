### Additional Experiments: Conversational LLM Generalization

**Table 1: LLM triage with/without relevance cues for Claude and Gemini. Metrics (except IFA) in %. Interpretability shown as with no relevance cues $\rightarrow$ with relevance cues.**

| Dataset | Acc | F1 (F/T) | T@1/3/5 | IFA | R@T20 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Claude** | | | | | |
| JBMC | 93.0 | 90.1/94.4 | 27/95/97 $\rightarrow$ 27/95/97 | 1.1 $\rightarrow$ 1.1 | 57.2 $\rightarrow$ 61.5 |
| Jayhorn | 95.0 | 92.9/96.2 | 11/96/99 $\rightarrow$ 11/96/99 | 1.1 $\rightarrow$ 1.1 | 53.5 $\rightarrow$ 57.2 |
| CBMC-A | 74.0 | 39.5/83.2 | 37/74/88 $\rightarrow$ 39/74/88 | 2.2 $\rightarrow$ 2.1 | 45.1 $\rightarrow$ 47.3 |
| CBMC-B | 73.4 | 79.2/62.9 | 37/76/89 $\rightarrow$ 40/76/89 | 1.8 $\rightarrow$ 1.7 | 45.8 $\rightarrow$ 47.6 |
| **Gemini** | | | | | |
| JBMC | 87.4 | 83.7/89.7 | 32/53/63 $\rightarrow$ 38/61/71 | 2.5 $\rightarrow$ 2.1 | 37.2 $\rightarrow$ 39.5 |
| Jayhorn | 88.7 | 84.7/91.1 | 26/46/61 $\rightarrow$ 33/58/71 | 2.8 $\rightarrow$ 2.4 | 35.2 $\rightarrow$ 36.5 |
| CBMC-A | 67.0 | 34.1/77.3 | 38/60/72 $\rightarrow$ 38/74/88 | 3.1 $\rightarrow$ 2.2 | 37.2 $\rightarrow$ 46.7 |
| CBMC-B | 83.3 | 89.4/58.3 | 31/57/61 $\rightarrow$ 38/75/88 | 3.9 $\rightarrow$ 1.8 | 31.7 $\rightarrow$ 49.7 |

**Insights from Claude and Gemini Evaluations:**
* **Classification Performance:** Claude achieved strong classification performance on Java (≈93-95%), while Gemini maintained a solid lead at ≈87-88%. On C datasets, Gemini reached a competitive 83.3% on CBMC-B.
* **Response to Relevance Cues:** Gemini exhibited steady, incremental improvements in Top-$k$ hit rates and IFA across the board. Claude maintained highly stable Top-$k$ and IFA metrics regardless of cues on Java, but showed minor improvements on C datasets.
* **Recall/Budget Trade-offs:** Unlike GPT-5, Recall@Top20% consistently increased across *both* Java and C datasets for both Claude and Gemini when relevance cues were provided. This suggests these models leverage the cues without overly hyper-focusing on them at the expense of adjacent relevant code.
