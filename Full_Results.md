# Tables (Markdown)

## Table 1 — RQ1/RQ2 Interpretability (IFA, R@T%)

*Cells are reported as mean<sub>std</sub>.*

| Model | App | JBMC IFA ↓ | JBMC R@T% ↑ | Jayhorn IFA ↓ | Jayhorn R@T% ↑ | CBMC-A IFA ↓ | CBMC-A R@T% ↑ | CBMC-B IFA ↓ | CBMC-B R@T% ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek | Base | 2.3<sub>2.1</sub> | 23.7<sub>23.9</sub> | 1.9<sub>1.3</sub> | 34.4<sub>23.7</sub> | 2.2<sub>2.3</sub> | 39.9<sub>21.9</sub> | 2.1<sub>1.8</sub> | 39.6<sub>20.8</sub> |
|  | KL | 1.2<sub>2.5</sub> | 47.1<sub>33.2</sub> | 1.0<sub>1.6</sub> | 48.7<sub>29.7</sub> | 1.2<sub>1.8</sub> | 54.3<sub>18.6</sub> | 1.3<sub>1.4</sub> | 47.5<sub>20.4</sub> |
|  | EGL | 2.0<sub>2.6</sub> | 41.1<sub>36.3</sub> | 1.7<sub>1.6</sub> | 36.9<sub>33.0</sub> | 2.1<sub>0.8</sub> | 54.2<sub>17.3</sub> | 0.9<sub>1.6</sub> | 56.3<sub>20.2</sub> |
| CodeLlama | Base | 2.2<sub>2.4</sub> | 27.7<sub>24.2</sub> | 1.9<sub>1.4</sub> | 31.7<sub>21.0</sub> | 8.7<sub>9.0</sub> | 39.9<sub>18.7</sub> | 6.1<sub>8.8</sub> | 37.3<sub>19.0</sub> |
|  | KL | 1.4<sub>2.4</sub> | 37.0<sub>32.4</sub> | 0.9<sub>1.1</sub> | 52.8<sub>30.8</sub> | 5.9<sub>6.2</sub> | 46.9<sub>22.7</sub> | 4.5<sub>3.6</sub> | 42.5<sub>21.4</sub> |
|  | EGL | 1.9<sub>2.4</sub> | 37.2<sub>31.2</sub> | 1.4<sub>2.4</sub> | 43.7<sub>30.6</sub> | 6.2<sub>8.4</sub> | 42.6<sub>17.1</sub> | 1.8<sub>2.2</sub> | 47.9<sub>18.6</sub> |
| WaveCoder | Base | 2.3<sub>2.2</sub> | 26.6<sub>24.6</sub> | 1.9<sub>1.5</sub> | 33.5<sub>22.1</sub> | 4.3<sub>6.2</sub> | 34.5<sub>16.6</sub> | 3.8<sub>8.3</sub> | 35.1<sub>20.1</sub> |
|  | KL | 2.0<sub>3.6</sub> | 42.5<sub>33.0</sub> | 1.5<sub>1.3</sub> | 38.2<sub>28.8</sub> | 3.7<sub>4.3</sub> | 42.3<sub>21.8</sub> | 2.9<sub>7.8</sub> | 41.2<sub>18.5</sub> |
|  | EGL | 1.4<sub>2.2</sub> | 38.7<sub>28.7</sub> | 1.1<sub>1.0</sub> | 50.4<sub>29.5</sub> | 1.8<sub>2.3</sub> | 45.0<sub>17.6</sub> | 3.3<sub>3.2</sub> | 42.7<sub>19.3</sub> |
| CodeGemma | Base | 2.3<sub>2.7</sub> | 32.3<sub>28.5</sub> | 2.0<sub>1.6</sub> | 32.2<sub>23.9</sub> | 1.8<sub>2.4</sub> | 40.8<sub>18.7</sub> | 10.3<sub>11.5</sub> | 36.7<sub>19.0</sub> |
|  | KL | 2.1<sub>2.0</sub> | 32.1<sub>31.8</sub> | 2.0<sub>2.8</sub> | 37.9<sub>26.9</sub> | 1.3<sub>1.7</sub> | 48.7<sub>17.1</sub> | 1.9<sub>1.6</sub> | 47.5<sub>21.3</sub> |
|  | EGL | 1.9<sub>2.6</sub> | 34.4<sub>30.2</sub> | 1.2<sub>1.3</sub> | 48.8<sub>30.7</sub> | 1.5<sub>3.5</sub> | 49.0<sub>17.9</sub> | 2.6<sub>3.2</sub> | 45.6<sub>20.5</sub> |
| Stable-Code | Base | 2.1<sub>2.0</sub> | 32.6<sub>28.2</sub> | 2.1<sub>1.6</sub> | 33.5<sub>26.0</sub> | 4.5<sub>5.7</sub> | 41.2<sub>18.0</sub> | 4.7<sub>4.5</sub> | 36.9<sub>17.5</sub> |
|  | KL | 2.0<sub>2.3</sub> | 36.9<sub>29.1</sub> | 1.0<sub>1.1</sub> | 49.0<sub>29.4</sub> | 3.2<sub>3.7</sub> | 46.1<sub>15.7</sub> | 2.0<sub>2.3</sub> | 43.6<sub>20.3</sub> |
|  | EGL | 1.4<sub>2.5</sub> | 45.6<sub>34.2</sub> | 1.0<sub>1.6</sub> | 54.6<sub>31.9</sub> | 2.7<sub>2.6</sub> | 52.9<sub>18.6</sub> | 2.7<sub>2.1</sub> | 51.1<sub>19.7</sub> |
| Qwen2.5 | Base | 2.2<sub>2.2</sub> | 35.3<sub>30.2</sub> | 2.1<sub>1.6</sub> | 29.1<sub>23.9</sub> | 3.5<sub>3.4</sub> | 35.2<sub>16.5</sub> | 4.1<sub>8.1</sub> | 37.2<sub>21.3</sub> |
|  | KL | 1.6<sub>2.3</sub> | 39.8<sub>32.2</sub> | 1.4<sub>1.1</sub> | 46.2<sub>29.7</sub> | 2.3<sub>3.0</sub> | 52.9<sub>18.9</sub> | 1.7<sub>2.3</sub> | 51.6<sub>19.6</sub> |
|  | EGL | 1.6<sub>2.5</sub> | 39.8<sub>34.3</sub> | 1.4<sub>1.4</sub> | 39.0<sub>28.7</sub> | 3.2<sub>4.1</sub> | 45.7<sub>18.5</sub> | 3.9<sub>2.8</sub> | 43.9<sub>21.1</sub> |
| StarCoder2 | Base | 2.3<sub>2.6</sub> | 26.2<sub>22.5</sub> | 2.6<sub>1.9</sub> | 26.6<sub>26.0</sub> | 4.2<sub>8.3</sub> | 39.7<sub>18.4</sub> | 4.4<sub>5.3</sub> | 34.5<sub>19.1</sub> |
|  | KL | 1.9<sub>2.7</sub> | 42.4<sub>35.1</sub> | 1.9<sub>2.1</sub> | 33.1<sub>27.6</sub> | 2.5<sub>6.1</sub> | 55.0<sub>18.2</sub> | 1.8<sub>2.9</sub> | 49.3<sub>19.7</sub> |
|  | EGL | 2.4<sub>2.1</sub> | 34.9<sub>28.8</sub> | 1.4<sub>1.8</sub> | 43.4<sub>32.6</sub> | 1.6<sub>4.0</sub> | 47.5<sub>19.3</sub> | 2.7<sub>4.1</sub> | 47.9<sub>19.6</sub> |
| NLM | Base | 2.5<sub>3.7</sub> | 27.1<sub>26.7</sub> | 2.6<sub>7.3</sub> | 25.3<sub>22.4</sub> | 8.1<sub>12.0</sub> | 38.3<sub>17.1</sub> | 7.9<sub>8.7</sub> | 36.5<sub>21.1</sub> |
|  | KL | 2.2<sub>4.4</sub> | 43.0<sub>31.4</sub> | 1.8<sub>1.4</sub> | 37.3<sub>29.3</sub> | 4.7<sub>7.4</sub> | 45.1<sub>20.9</sub> | 4.7<sub>6.3</sub> | 38.6<sub>19.9</sub> |
|  | EGL | 2.1<sub>3.0</sub> | 40.1<sub>31.7</sub> | 2.0<sub>2.0</sub> | 35.3<sub>28.8</sub> | 4.3<sub>8.6</sub> | 47.3<sub>21.8</sub> | 3.5<sub>3.7</sub> | 41.0<sub>19.9</sub> |
| LSTM | Base | 2.4<sub>4.5</sub> | 29.3<sub>25.3</sub> | 2.4<sub>4.9</sub> | 30.5<sub>25.6</sub> | 6.9<sub>12.7</sub> | 40.6<sub>20.8</sub> | 8.0<sub>14.8</sub> | 36.0<sub>19.9</sub> |
|  | KL | 1.9<sub>2.9</sub> | 46.6<sub>34.2</sub> | 1.7<sub>3.9</sub> | 50.7<sub>32.2</sub> | 4.0<sub>7.7</sub> | 46.8<sub>22.3</sub> | 3.5<sub>10.3</sub> | 42.6<sub>21.8</sub> |
|  | EGL | 1.9<sub>3.3</sub> | 49.0<sub>31.8</sub> | 1.7<sub>1.8</sub> | 45.9<sub>31.2</sub> | 3.7<sub>3.4</sub> | 49.3<sub>21.3</sub> | 2.6<sub>9.6</sub> | 43.3<sub>23.1</sub> |

---

## Table 2 — Static Analysis Classification (Acc, F1(F), F1(T))

*Cells are reported as mean<sub>std</sub> (in %).*

### JBMC (Java)

| Model | App | Acc | F1(F) | F1(T) |
|---|---|---:|---:|---:|
| DeepSeek | Base | 74.6<sub>4.3</sub> | 58.3<sub>12.2</sub> | 81.2<sub>4.1</sub> |
|  | KL | **77.6**<sub>3.7</sub> | **60.5**<sub>14.3</sub> | **84.1**<sub>1.5</sub> |
|  | EGL | 74.6<sub>3.8</sub> | 53.6<sub>11.0</sub> | 82.4<sub>2.5</sub> |
| CodeLlama | Base | 74.6<sub>4.3</sub> | 59.5<sub>4.1</sub> | 81.3<sub>4.3</sub> |
|  | KL | **79.1**<sub>2.1</sub> | **69.4**<sub>3.9</sub> | **84.1**<sub>1.4</sub> |
|  | EGL | 74.7<sub>5.0</sub> | 63.2<sub>5.6</sub> | 83.5<sub>1.7</sub> |
| WaveCoder | Base | 72.5<sub>5.0</sub> | 44.2<sub>25.8</sub> | 79.5<sub>5.7</sub> |
|  | KL | 74.6<sub>4.3</sub> | 53.5<sub>13.2</sub> | 82.3<sub>2.4</sub> |
|  | EGL | 74.6<sub>5.5</sub> | 49.3<sub>28.3</sub> | 80.6<sub>6.1</sub> |
| CodeGemma | Base | 73.6<sub>5.0</sub> | 57.9<sub>9.5</sub> | 80.5<sub>4.3</sub> |
|  | KL | **77.3**<sub>5.9</sub> | **60.5**<sub>16.5</sub> | **83.8**<sub>3.7</sub> |
|  | EGL | 75.9<sub>3.0</sub> | 54.1<sub>14.8</sub> | 83.4<sub>1.1</sub> |
| Stable-Code | Base | 72.5<sub>3.9</sub> | 38.6<sub>23.7</sub> | 80.1<sub>5.6</sub> |
|  | KL | 75.2<sub>4.9</sub> | 57.8<sub>10.8</sub> | 82.4<sub>3.0</sub> |
|  | EGL | 74.9<sub>2.2</sub> | 52.5<sub>13.4</sub> | 82.6<sub>0.7</sub> |
| Qwen2.5 | Base | 70.9<sub>4.1</sub> | 40.7<sub>24.2</sub> | 78.0<sub>5.8</sub> |
|  | KL | 74.9<sub>2.2</sub> | 52.2<sub>13.3</sub> | 82.7<sub>0.5</sub> |
|  | EGL | 74.2<sub>3.9</sub> | 49.9<sub>12.6</sub> | 82.5<sub>2.5</sub> |
| StarCoder2 | Base | 72.9<sub>2.9</sub> | 53.5<sub>8.4</sub> | 80.5<sub>3.6</sub> |
|  | KL | 75.6<sub>2.8</sub> | 52.2<sub>13.0</sub> | 83.5<sub>1.1</sub> |
|  | EGL | 74.6<sub>1.7</sub> | 50.3<sub>10.9</sub> | 82.8<sub>0.6</sub> |
| NLM | Base | 72.9<sub>3.9</sub> | 52.9<sub>13.4</sub> | 81.0<sub>2.6</sub> |
|  | KL | 75.3<sub>2.7</sub> | 54.7<sub>12.8</sub> | 83.0<sub>1.3</sub> |
|  | EGL | 73.2<sub>4.1</sub> | 54.3<sub>27.4</sub> | 81.1<sub>1.8</sub> |
| LSTM | Base | 72.2<sub>4.1</sub> | 49.4<sub>13.6</sub> | 80.8<sub>1.6</sub> |
|  | KL | 74.6<sub>3.6</sub> | 53.8<sub>12.3</sub> | 81.5<sub>2.1</sub> |
|  | EGL | 74.6<sub>3.6</sub> | 51.6<sub>10.3</sub> | 82.8<sub>2.5</sub> |
| RF | Base | 71.2<sub>5.8</sub> | 45.2<sub>14.0</sub> | 80.5<sub>2.7</sub> |

### Jayhorn (Java)

| Model | App | Acc | F1(F) | F1(T) |
|---|---|---:|---:|---:|
| DeepSeek | Base | 75.4<sub>5.9</sub> | 60.8<sub>10.8</sub> | 81.9<sub>4.5</sub> |
|  | KL | **78.5**<sub>4.7</sub> | **66.7**<sub>9.2</sub> | **84.0**<sub>3.0</sub> |
|  | EGL | **77.4**<sub>5.6</sub> | **63.4**<sub>12.6</sub> | 83.6<sub>3.4</sub> |
| CodeLlama | Base | 74.9<sub>4.2</sub> | 57.6<sub>11.6</sub> | 82.0<sub>2.1</sub> |
|  | KL | **78.5**<sub>2.9</sub> | **65.1**<sub>6.5</sub> | **84.3**<sub>2.0</sub> |
|  | EGL | 75.4<sub>4.3</sub> | 58.7<sub>11.9</sub> | 82.3<sub>2.1</sub> |
| WaveCoder | Base | 70.8<sub>5.3</sub> | 56.9<sub>5.8</sub> | 77.4<sub>5.8</sub> |
|  | KL | 73.3<sub>3.9</sub> | 54.9<sub>7.1</sub> | 80.9<sub>3.4</sub> |
|  | EGL | 73.3<sub>4.7</sub> | 57.9<sub>6.7</sub> | 80.3<sub>4.3</sub> |
| CodeGemma | Base | 73.3<sub>3.4</sub> | 50.8<sub>7.2</sub> | 81.7<sub>2.2</sub> |
|  | KL | 75.9<sub>3.9</sub> | 60.8<sub>10.3</sub> | 82.5<sub>2.0</sub> |
|  | EGL | 74.9<sub>2.8</sub> | 57.3<sub>5.9</sub> | 82.2<sub>2.0</sub> |
| Stable-Code | Base | 72.3<sub>8.0</sub> | 58.2<sub>11.5</sub> | 79.0<sub>6.9</sub> |
|  | KL | 76.4<sub>4.9</sub> | 61.0<sub>10.7</sub> | 83.0<sub>3.2</sub> |
|  | EGL | 75.4<sub>2.9</sub> | 58.4<sub>6.6</sub> | 82.5<sub>2.0</sub> |
| Qwen2.5 | Base | 70.3<sub>6.4</sub> | 57.6<sub>4.9</sub> | 76.8<sub>6.4</sub> |
|  | KL | 75.9<sub>4.3</sub> | 59.8<sub>12.1</sub> | 82.6<sub>2.1</sub> |
|  | EGL | 75.9<sub>4.3</sub> | 59.8<sub>12.1</sub> | 82.6<sub>2.1</sub> |
| StarCoder2 | Base | 72.8<sub>6.4</sub> | 61.7<sub>7.9</sub> | 78.7<sub>6.0</sub> |
|  | KL | **76.9**<sub>3.6</sub> | **62.9**<sub>7.0</sub> | 83.2<sub>2.3</sub> |
|  | EGL | 75.4<sub>5.0</sub> | 56.5<sub>12.9</sub> | 82.7<sub>2.7</sub> |
| NLM | Base | 70.9<sub>5.7</sub> | 42.7<sub>7.7</sub> | 80.5<sub>2.3</sub> |
|  | KL | 71.8<sub>3.6</sub> | 47.6<sub>8.3</sub> | 80.7<sub>3.1</sub> |
|  | EGL | 71.8<sub>3.6</sub> | 47.6<sub>8.3</sub> | 80.7<sub>3.1</sub> |
| LSTM | Base | 72.9<sub>6.1</sub> | 55.6<sub>10.4</sub> | 80.5<sub>2.5</sub> |
|  | KL | 74.4<sub>5.3</sub> | 50.0<sub>6.1</sub> | 82.8<sub>1.3</sub> |
|  | EGL | 71.8<sub>3.5</sub> | 56.0<sub>13.5</sub> | 79.3<sub>1.5</sub> |
| RF | Base | 65.1<sub>6.6</sub> | 45.2<sub>10.1</sub> | 74.4<sub>5.9</sub> |

### CBMC-A (C)

| Model | App | Acc | F1(F) | F1(T) |
|---|---|---:|---:|---:|
| DeepSeek | Base | 79.2<sub>4.4</sub> | 69.8<sub>6.5</sub> | 84.0<sub>3.6</sub> |
|  | KL | **81.5**<sub>2.5</sub> | 72.5<sub>5.5</sub> | **86.0**<sub>1.6</sub> |
|  | EGL | 80.5<sub>3.0</sub> | 70.5<sub>6.5</sub> | 85.4<sub>1.9</sub> |
| CodeLlama | Base | 77.1<sub>3.1</sub> | 65.1<sub>6.7</sub> | 82.9<sub>1.9</sub> |
|  | KL | 78.8<sub>3.2</sub> | 68.2<sub>7.5</sub> | 84.0<sub>1.8</sub> |
|  | EGL | 79.2<sub>2.7</sub> | 68.2<sub>8.0</sub> | 84.5<sub>2.3</sub> |
| WaveCoder | Base | 79.4<sub>1.4</sub> | 69.5<sub>1.2</sub> | 84.4<sub>1.3</sub> |
|  | KL | **81.7**<sub>1.9</sub> | 72.6<sub>5.0</sub> | **86.3**<sub>2.0</sub> |
|  | EGL | 80.7<sub>1.8</sub> | **73.9**<sub>5.1</sub> | **87.1**<sub>1.8</sub> |
| CodeGemma | Base | 77.1<sub>4.9</sub> | 71.7<sub>4.8</sub> | 79.0<sub>9.7</sub> |
|  | KL | 80.0<sub>3.6</sub> | 73.1<sub>4.3</sub> | 85.3<sub>3.1</sub> |
|  | EGL | 80.3<sub>4.2</sub> | 72.9<sub>7.2</sub> | 85.7<sub>4.1</sub> |
| Stable-Code | Base | 78.7<sub>1.8</sub> | 68.5<sub>4.2</sub> | 83.9<sub>1.1</sub> |
|  | KL | 80.8<sub>2.1</sub> | **75.7**<sub>4.0</sub> | **86.7**<sub>2.6</sub> |
|  | EGL | 80.8<sub>2.1</sub> | **75.7**<sub>4.0</sub> | **86.7**<sub>2.6</sub> |
| Qwen2.5 | Base | 78.5<sub>1.6</sub> | 67.9<sub>3.2</sub> | 83.8<sub>1.2</sub> |
|  | KL | 80.8<sub>1.8</sub> | 73.0<sub>2.6</sub> | 85.1<sub>1.4</sub> |
|  | EGL | 80.4<sub>1.8</sub> | **73.2**<sub>2.3</sub> | 85.2<sub>1.5</sub> |
| StarCoder2 | Base | 73.0<sub>2.8</sub> | 58.1<sub>8.7</sub> | 79.8<sub>1.9</sub> |
|  | KL | 75.0<sub>1.6</sub> | 64.4<sub>3.0</sub> | 80.7<sub>1.5</sub> |
|  | EGL | 75.0<sub>1.6</sub> | 64.4<sub>3.0</sub> | 80.7<sub>1.5</sub> |
| NLM | Base | 73.9<sub>3.5</sub> | 60.0<sub>2.1</sub> | 80.6<sub>1.2</sub> |
|  | KL | 74.9<sub>3.2</sub> | 61.5<sub>5.6</sub> | 81.3<sub>1.1</sub> |
|  | EGL | 73.9<sub>3.5</sub> | 60.0<sub>2.1</sub> | 80.6<sub>1.2</sub> |
| LSTM | Base | 74.4<sub>2.1</sub> | 61.1<sub>1.2</sub> | 80.9<sub>1.6</sub> |
|  | KL | 76.4<sub>3.9</sub> | 63.0<sub>6.7</sub> | 82.7<sub>1.4</sub> |
|  | EGL | 74.9<sub>3.0</sub> | 61.5<sub>5.0</sub> | 81.3<sub>1.2</sub> |
| RF | Base | 71.2<sub>3.9</sub> | 56.2<sub>4.9</sub> | 78.5<sub>2.1</sub> |

### CBMC-B (C)

| Model | App | Acc | F1(F) | F1(T) |
|---|---|---:|---:|---:|
| DeepSeek | Base | 76.1<sub>2.7</sub> | 81.1<sub>2.4</sub> | 66.2<sub>8.1</sub> |
|  | KL | 77.4<sub>1.9</sub> | **81.9**<sub>1.8</sub> | 68.5<sub>9.7</sub> |
|  | EGL | **77.7**<sub>2.5</sub> | **82.8**<sub>1.0</sub> | 67.5<sub>9.6</sub> |
| CodeLlama | Base | 73.2<sub>1.9</sub> | 77.8<sub>1.9</sub> | 66.0<sub>3.2</sub> |
|  | KL | 76.1<sub>1.2</sub> | 81.0<sub>1.2</sub> | 67.6<sub>1.4</sub> |
|  | EGL | 76.1<sub>1.2</sub> | 81.0<sub>1.2</sub> | 67.6<sub>1.4</sub> |
| WaveCoder | Base | 75.2<sub>4.2</sub> | 80.8<sub>4.0</sub> | 63.9<sub>8.1</sub> |
|  | KL | 77.1<sub>4.0</sub> | 82.0<sub>3.6</sub> | 68.7<sub>4.1</sub> |
|  | EGL | 77.1<sub>4.0</sub> | 82.0<sub>3.6</sub> | 68.7<sub>4.1</sub> |
| CodeGemma | Base | 74.3<sub>3.8</sub> | 78.0<sub>4.0</sub> | 68.7<sub>3.6</sub> |
|  | KL | **78.2**<sub>3.0</sub> | 82.7<sub>2.3</sub> | 68.9<sub>10.2</sub> |
|  | EGL | **77.6**<sub>3.5</sub> | **83.3**<sub>1.8</sub> | 64.4<sub>12.1</sub> |
| Stable-Code | Base | 75.5<sub>1.7</sub> | 79.4<sub>2.1</sub> | 69.5<sub>2.8</sub> |
|  | KL | **79.0**<sub>1.5</sub> | **83.4**<sub>2.1</sub> | **71.2**<sub>3.3</sub> |
|  | EGL | **78.5**<sub>1.7</sub> | 82.6<sub>2.5</sub> | **71.7**<sub>4.0</sub> |
| Qwen2.5 | Base | 75.3<sub>2.3</sub> | 79.9<sub>2.9</sub> | 67.9<sub>1.4</sub> |
|  | KL | **77.6**<sub>2.7</sub> | 82.2<sub>2.1</sub> | 69.6<sub>3.8</sub> |
|  | EGL | **78.1**<sub>2.5</sub> | 82.6<sub>2.0</sub> | **70.3**<sub>3.5</sub> |
| StarCoder2 | Base | 73.1<sub>1.6</sub> | 77.2<sub>1.9</sub> | 67.0<sub>1.3</sub> |
|  | KL | 76.1<sub>2.4</sub> | 80.8<sub>3.9</sub> | 68.1<sub>1.7</sub> |
|  | EGL | 76.1<sub>2.4</sub> | 80.8<sub>3.9</sub> | 68.1<sub>1.7</sub> |
| NLM | Base | 72.2<sub>2.6</sub> | 78.4<sub>2.9</sub> | 61.1<sub>3.9</sub> |
|  | KL | 75.9<sub>2.0</sub> | 81.2<sub>2.4</sub> | 66.7<sub>3.5</sub> |
|  | EGL | 73.7<sub>3.2</sub> | 79.8<sub>2.8</sub> | 62.4<sub>5.9</sub> |
| LSTM | Base | 72.9<sub>2.8</sub> | 78.6<sub>2.8</sub> | 63.3<sub>2.1</sub> |
|  | KL | 75.8<sub>2.9</sub> | 81.3<sub>2.8</sub> | 65.5<sub>3.7</sub> |
|  | EGL | 74.4<sub>3.3</sub> | 80.5<sub>2.2</sub> | 63.0<sub>7.9</sub> |
| RF | Base | 69.9<sub>4.0</sub> | 76.8<sub>3.8</sub> | 55.9<sub>7.0</sub> |
