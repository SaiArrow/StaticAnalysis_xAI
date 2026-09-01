# Understanding and Improving ML-based Static Analysis Result Classification via Explainable AI

## Datasets
Static Analysis Result Dataset (per tool)

C: ['/c/CBMC-A.csv', '/c/CBMC-B.csv']

Java: ['/java/jayhorn.csv', '/java/jbmc.csv']



Source Code Relevancy Dataset (Tool Agnostic)

C: ['/c/relevancy.json']

Java: ['/java/relevancy.json']


## Model Training
Code for training the models has been included within each folder, The code files required for training the models can be downloaded from the [SV-COMP repository](https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks)

RQ1 Models can be created using the model.py file within each folder

RQ2 Models can be created using the kl_model.py and egl_model.py file within each folder

For RQ3, we use the following prompt,

'You're a static analyzer tool expert, I am gonna give you csv files for jbmc and jayhorn, they contain a file name and corresponding analysis label done by the tool, for example, in test_jayhorn.csv for file ExSymExe15_true.java the label is False meaning that Jayhorn tool thinks that there is an issue/bug within the code.

After this I am gonna give you a list of code files, if the file name exists in the csv file, analyze it and if you disagree with the Tool analysis give it a label 0 for Incorrect and 1 for Correct. If the file exists in both CSV files, give me 2 outputs one for each tool. Along with the label, also give me a list of ranking of lines in the code sorted by relevance (most relevant lines first), relevancy is dictated by the parts of the code that a developer should look at to decide if there is a bug or not. All lines containing code should be included in the list and ranked.

For example for the code file "ExSymExe15_true.java", since it exists in both JBMC and Jayhorn csv files, give me the output,

{"file":"ExSymExe15_true.java", "tool":"JBMC", "label":0, "ranked_relevant_lines":[53, 51, 40, 39, 34, 27, 29, 30, 31, 33, 35, 37, 38, 42, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 60, 61, 62]}
{"file":"ExSymExe15_true.java", "tool":"Jayhorn", "label":0, "ranked_relevant_lines":[53, 51, 40, 39, 34, 27, 29, 30, 31, 33, 35, 37, 38, 42, 45, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 58, 60, 61, 62]}

Label is 0 because both tools think there is a Bug (analysis_label is False in both csv file) but there isn't a bug so the analysis is incorrect. Lines 53, 51, 40, 39 and 34 are the most relevant lines in the code hence they are ranked early.`

## Tool Configs
JBMC Configuration: `--drop-unused-functions --full-slice --java-threading --no-pretty-names --refine --string-printable --depth 5000 --unwind 5 --max-nondet-array-length 10 --max-nondet-string-length 100 --max-nondet-tree-depth 2000 --java-max-vla-length 100 --arrays-uf-always --reachability-slice-fb`

Jayhorn Configuration: `-rta -specs -bounded-heap-size 10 -heap-limit 1 -heap-mode bounded -initial-heap-size 1 -inline-count -1 -inline-size 100 -mem-prec 1 -solver eldarica -step-heap-size 1 -solver-options debug -solution`

CBMC-A Configuration: `--no-assumptions --no-pretty-names --refine --refine-strings --depth 100 --unwind 10 --max-nondet-tree-depth 100 --min-null-tree-depth 100 --paths fifo --mm sc --arrays-uf-never --reachability-slice-fb --round-to-plus-inf --cprover-smt2`

CBMC-B Configuration: `--drop-unused-functions --full-slice --refine --slice-formula --unwind 1 --max-nondet-tree-depth 2000 --min-null-tree-depth 10 --mm pso --reachability-slice --round-to-plus-inf --cprover-smt2`
