# Understanding and Improving ML-based Static Analysis Result Classification via Explainable AI

Artifact for the ICST 2026 paper by S. Yerramreddy, M. Rafieian, S. Wei, and A. Porter.

📄 [Paper](https://ieeexplore.ieee.org/abstract/document/11600526) · 📊 [Full results](Full_Results.md) · 🗂️ [Relevance annotations](#source-code-relevancy-dataset)

Static analysis tools report far more warnings than developers can review, and most of them are false positives. Machine-learned classifiers can filter that stream, but a classifier that reaches high accuracy while attending to the wrong lines of code is not much help to the developer who has to act on its output. This repository contains the dataset, models, and evaluation code we used to
measure that gap and to narrow it.

**Main result.** Relevance-guided training raises the top-1 relevant-line hit rate from 6–35% to 12–63% across four analyzer configurations, while also improving classification accuracy.

![Relevance before and after guided training](motivation_before_after.pdf)

---

## What's here

| Path | Contents |
|---|---|
| `c/` | CBMC-A and CBMC-B data, model training files, and relevance annotations |
| `java/` | JBMC and Jayhorn data, model training files, and relevance annotations |
| `Full_Results.md` | Per-tool results for every configuration in the paper |
| `Significance_tests.md` | Statistical tests backing the reported comparisons |
| `Additional_LLMs.md` | RQ3 results for the LLM baselines |
| `*_TopK.pdf` | Top-k relevance curves per analyzer |


## Datasets

### Static analysis result dataset

One CSV per analyzer configuration, mapping each benchmark file to the tool's verdict and the ground-truth label:

- **C** — `c/CBMC-A.csv`, `c/CBMC-B.csv`
- **Java** — `java/jbmc.csv`, `java/jayhorn.csv`

### Source code relevancy dataset

Line-level relevance annotations, collected independently of any analyzer, marking the lines a
developer needs to read to decide whether a warning is real. To our knowledge this is the first
line-level relevance-annotated dataset for static analysis triage.

- `c/relevant.json`, `java/relevant.json`


## Models

Each language directory contains the same three training entry points:

| File | Research question | Description |
|---|---|---|
| `model.py` | RQ1 | Baseline Transformer classifiers, no relevance signal |
| `kl_model.py` | RQ2 | Prediction consistency under masking of irrelevant lines |
| `egl_model.py` | RQ2 | Explanation-guided learning against the relevance annotations |

`nn.py`, `nn_kl.py`, and `nn_egl.py` holds the code for the language neural network equivalent.

RQ3 evaluates LLMs as zero-shot triage classifiers. The exact prompt is as follows:

`You're a static analyzer tool expert, I am gonna give you csv files for jbmc and jayhorn, they contain a file name and corresponding analysis label done by the tool, for example, in test_jayhorn.csv for file ExSymExe15_true.java the label is False meaning that Jayhorn tool thinks that there is an issue/bug within the code.`

`After this I am gonna give you a list of code files, if the file name exists in the csv file, analyze it and if you disagree with the Tool analysis give it a label 0 for Incorrect and 1 for Correct. If the file exists in both CSV files, give me 2 outputs one for each tool. Along with the label, also give me a list of ranking of lines in the code sorted by relevance (most relevant lines first), relevancy is dictated by the parts of the code that a developer should look at to decide if there is a bug or not. All lines containing code should be included in the list and ranked.`

`For example for the code file "ExSymExe15_true.java", since it exists in both JBMC and Jayhorn csv files, give me the output,`

`{"file":"ExSymExe15_true.java", "tool":"JBMC", "label":0, "ranked_relevant_lines":[53, 51, 40, 39, 34, 27, 29, 30, 31, 33, 35, 37, 38, 42, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 60, 61, 62]} {"file":"ExSymExe15_true.java", "tool":"Jayhorn", "label":0, "ranked_relevant_lines":[53, 51, 40, 39, 34, 27, 29, 30, 31, 33, 35, 37, 38, 42, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 60, 61, 62]}`

`Label is 0 because both tools think there is a Bug (analysis_label is False in both csv file) but there isn't a bug so the analysis is incorrect. Lines 53, 51, 40, 39 and 34 are the most relevant lines in the code hence they are ranked early.`

## Analyzer configurations

The benchmarks were generated with the following tool settings. These matter — as we show in the
paper, configuration choices move classifier accuracy substantially, so results are not comparable
across differently configured runs.

<details>
<summary>JBMC</summary>

```
--drop-unused-functions --full-slice --java-threading --no-pretty-names --refine
--string-printable --depth 5000 --unwind 5 --max-nondet-array-length 10
--max-nondet-string-length 100 --max-nondet-tree-depth 2000 --java-max-vla-length 100
--arrays-uf-always --reachability-slice-fb
```
</details>

<details>
<summary>Jayhorn</summary>

```
-rta -specs -bounded-heap-size 10 -heap-limit 1 -heap-mode bounded -initial-heap-size 1
-inline-count -1 -inline-size 100 -mem-prec 1 -solver eldarica -step-heap-size 1
-solver-options debug -solution
```
</details>

<details>
<summary>CBMC-A</summary>

```
--no-assumptions --no-pretty-names --refine --refine-strings --depth 100 --unwind 10
--max-nondet-tree-depth 100 --min-null-tree-depth 100 --paths fifo --mm sc
--arrays-uf-never --reachability-slice-fb --round-to-plus-inf --cprover-smt2
```
</details>

<details>
<summary>CBMC-B</summary>

```
--drop-unused-functions --full-slice --refine --slice-formula --unwind 1
--max-nondet-tree-depth 2000 --min-null-tree-depth 10 --mm pso --reachability-slice
--round-to-plus-inf --cprover-smt2
```
</details>

Benchmark programs come from the
[SV-COMP repository](https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks).

## Citing

```bibtex
@inproceedings{yerramreddy2026understanding,
  author={Yerramreddy, Sai and Rafieian, Mohammad and Wei, Shiyi and Porter, Adam},
  booktitle={2026 IEEE International Conference on Software Testing, Verification and Validation (ICST)}, 
  title={Understanding and Improving ML-based Static Analysis Result Classification via Explainable AI}, 
  year={2026},
  pages={261-272},
  doi={10.1109/ICST69053.2026.00043}
}
```

## License

MIT. See [LICENSE](LICENSE).
