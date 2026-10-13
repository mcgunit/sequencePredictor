# Small Models for Time-Series Forecasting & Anomaly Detection — Research Roadmap

> Status: **Draft outline** · Owner: Michael · Last updated: 2026-10-13

## 1. Goal

Build in-house knowledge and IP on **forecasting** and **anomaly detection** for time-series data using
small, CPU-friendly models, and compare fundamentally different approaches on the same benchmark:

1. Classic deep sequence models (LSTM / GRU / TCN / DeepAR)
2. Pretrained time-series foundation models (Chronos, TimesFM, ...)
3. Off-the-shelf small LLMs (0.5B–4B) via llama.cpp, zero-shot
4. Fine-tuned small LLMs
5. Our own tiny "LLM" trained from scratch (transformer, LSTM, RWKV, Mamba, xLSTM)
6. A hybrid: numeric model for scoring + small LLM for reasoning and explanation

### Constraints

| Constraint | Implication |
|---|---|
| CPU-optimized inference is the primary target | Track latency, RAM and throughput on CPU for every approach, not only accuracy |
| Consumer GPUs for training (reference: GTX 1060 6 GB) | Models up to ~50M params from scratch; LLM fine-tuning only via QLoRA on ≤ ~1.5B |
| Hyperparameter search is expensive (previously ~1 week with Optuna) | Use pruning, smaller search spaces, and multi-fidelity search (see §5.1) |
| Results must become company IP | Check model and dataset licenses before building on them (see §10) |

## 2. Task definitions

Both tasks share the same input; only the expected output differs.

**Input:** a history of rows `datetime, value1, ..., valueX` at a fixed interval
(15 min, hourly or daily), spanning up to a year (or 10–20 years for daily data).

**Task F — Forecasting:** predict the next value(s) at the same interval as the history.
- F1: one-step-ahead point forecast
- F2: multi-step horizon (e.g. next 24 h, next 7 days)
- F3: probabilistic forecast (quantiles / prediction interval)

**Task A — Anomaly detection:** decide whether the latest point (or a window) is anomalous.
- A1: point anomaly — is the last value anomalous?
- A2: window scoring — anomaly score per point over a window
- A3: explanation — why is it anomalous (hybrid / LLM tracks only)

## 3. Phase 0 — Shared evaluation harness (build first)

Every track plugs into the same harness so results are directly comparable.

### 3.1 Data
- Collect 3–5 representative datasets covering each granularity (15 min, hourly, daily), uni- and multivariate.
- Add 1–2 public benchmarks for external comparison (e.g. from the Monash archive or GIFT-Eval).
- **Anomaly labels:** real labels are rare, so inject synthetic anomalies (spikes, level shifts,
  trend changes, missing seasonality, flatlines) into clean series with known positions.

### 3.2 Splitting
- Rolling-origin (expanding window) backtesting — never random splits.
- Fixed test periods shared across all tracks.

### 3.3 Metrics
| Task | Metrics |
|---|---|
| Forecast (point) | MAE, RMSE, MASE (scale-free, compares to seasonal naive), sMAPE |
| Forecast (probabilistic) | CRPS / weighted quantile loss, interval coverage |
| Anomaly | Precision, recall, F1, detection delay; report point-adjusted F1 separately (it inflates scores) |
| Cost | CPU latency per prediction, peak RAM, model size on disk, training time, tokens/s (LLM tracks) |

### 3.4 Baselines (mandatory)
- Seasonal naive (value one season ago)
- ETS and ARIMA (e.g. `statsforecast`)
- STL decomposition + residual threshold (anomaly baseline)
- Isolation Forest on windowed features (anomaly baseline)

### 3.5 Output
- One results table per dataset × task, stored as CSV + generated Markdown report.

## 4. Proposed repository structure

```
ts-small-models/
├── README.md                  # Overview, how to run, results summary
├── harness/                   # Phase 0: data, splits, metrics, baselines, report
│   └── README.md
├── tracks/
│   ├── a_classic/             # LSTM, GRU, TCN, DeepAR
│   ├── b_foundation/          # Chronos, TimesFM, TiRex, ...
│   ├── c_llm_zeroshot/        # llama.cpp + prompt/serialization strategies
│   ├── d_llm_finetune/        # QLoRA fine-tuning + GGUF export
│   ├── e_tiny_lm/             # Own models from scratch
│   └── f_hybrid/              # Numeric scorer + LLM explainer
│       (each track has its own README.md with setup, results and findings)
├── data/                      # Not committed; documented in harness/README.md
└── results/                   # Generated tables and reports
```

## 5. Tracks

### 5.1 Track A — Classic deep sequence models

**What:** LSTM, GRU and TCN (existing experience), plus **DeepAR** (autoregressive LSTM producing a
probability distribution instead of a point value).

**Candidates:** LSTM, GRU, TCN, stacked combinations, DeepAR; optionally N-BEATS / N-HiTS as strong
non-recurrent references.

**Tooling:** PyTorch; GluonTS (DeepAR), Darts or NeuralForecast (many models behind one API).

**Making hyperparameter search cheaper:**
- Optuna pruning (`MedianPruner`, `HyperbandPruner`) to kill bad trials early
- Multi-fidelity: search on a subset of the data / fewer epochs, then retrain the top-k fully
- Narrower search spaces informed by earlier results
- Reuse best configs across datasets of the same granularity as warm starts

**Anomaly detection:** forecast-residual against the prediction interval (DeepAR gives this natively);
LSTM autoencoder reconstruction error as a second method.

**CPU deployment:** export to ONNX, int8 quantization, ONNX Runtime.

### 5.2 Track B — Time-series foundation models (zero-shot, then fine-tuned)

**What:** Models pretrained on large collections of time series that forecast unseen series zero-shot.
This is "the LLM recipe applied to numbers".

**Candidates:**
| Model | Notes |
|---|---|
| Chronos / Chronos-Bolt (Amazon) | T5-based; values scaled + quantized into tokens. Several sizes, small ones run fine on CPU |
| TimesFM (Google) | Decoder-only, patch-based |
| TiRex (NX-AI) | xLSTM-based forecaster — directly relevant to the xLSTM interest |
| Moirai (Salesforce) | Multivariate, any-frequency; **check weight license** |
| Lag-Llama | Open-source, Llama-style architecture for probabilistic forecasting |

**Experiments:**
1. Zero-shot on all harness datasets
2. Fine-tune the best 1–2 on our data
3. CPU benchmark (ONNX export where supported)

### 5.3 Track C — Off-the-shelf small LLMs, zero-shot (llama.cpp)

**What:** Can a 0.5B–4B general LLM forecast or detect anomalies from a serialized history?

**Candidate models:** Qwen (0.5B–4B), Gemma (small variants), Llama 3.2 1B/3B, SmolLM, Phi mini.

**Token budget (approx. 25 tokens per row with timestamp + 3 values):**
| Interval | Rows/year | Tokens |
|---|---|---|
| 15 min | 35,040 | ~875k |
| Hourly | 8,760 | ~220k |
| Daily | 365 | ~9k |

**Serialization strategies to compare:**
- S1 — Raw CSV rows (baseline)
- S2 — Compact: interval stated once, no timestamps, values scaled to fixed-precision integers (LLMTime-style)
- S3 — Multi-resolution: last N points at full resolution + daily aggregates (last month) + weekly aggregates (year)
- S4 — S3 + summary statistics (mean, std, min/max, same point last week / last year)

**llama.cpp techniques:**
- **GBNF grammar** to force output to be a valid number (or JSON) — eliminates parsing failures
- Sample N times at temperature > 0 and take the median → point forecast; spread → uncertainty
- KV-cache quantization and flash attention for longer contexts

**Anomaly variant:** same prompt, output `{"anomaly": bool, "score": 0-1, "reason": "..."}` via grammar.

### 5.4 Track D — Fine-tuned small LLMs (0.5B–2B)

**What:** Teach a small LLM the serialization format and the task.

**Approach:**
- QLoRA (Unsloth or PEFT); on 6 GB VRAM this is realistic up to ~1.5B, borderline at 2B
- Training data generated from the harness: (serialized history → next value / anomaly label) pairs,
  plus synthetic series (sine + trend + noise + injected anomalies) for volume
- Merge adapters → convert to GGUF → run on llama.cpp CPU

**Key question:** does fine-tuning close the gap to Track A/B, and at what inference cost?

**Reference (research only):** Time-LLM-style "reprogramming" adds numeric input layers to a frozen LLM.
Not llama.cpp-compatible, so lower priority for the CPU flow.

### 5.5 Track E — Our own tiny "LLM" from scratch

**What:** A small next-token model trained only on time-series tokens. This is the most IP-relevant track.

**Tokenization options:**
- T1 — Chronos-style: mean-scale the series, quantize into N bins (e.g. 4096) → one token per value
- T2 — Digit-level: fixed-precision digits as tokens (small vocabulary, longer sequences)
- T3 — Patches: groups of values embedded directly (not a classic tokenizer; PatchTST-style)

**Architectures:**
| Architecture | Notes | llama.cpp / GGUF possible? |
|---|---|---|
| Tiny transformer (Llama-style, nanoGPT-sized) | Easiest baseline | **Yes**, if built in the Llama architecture |
| LSTM / GRU language model | Builds on existing experience | No → ONNX Runtime |
| TCN | Fast, parallel training | No → ONNX Runtime |
| RWKV | RNN with transformer-style parallel training; constant memory per token | **Yes** (supported arch) |
| Mamba (state-space model) | Linear-time in sequence length; strong on long sequences | **Yes** (supported arch) |
| xLSTM | Modernized LSTM (exponential gating, matrix memory) | To verify |

**Why this matters for CPU flows:** recurrent/state-space models (RWKV, Mamba, xLSTM) have constant
memory and cost per new token, so long histories don't blow up RAM the way a transformer KV cache does.

**Scale on a GTX 1060:** 1M–50M parameters, trained on harness data + synthetic series.

**Deployment path:** if the model uses a llama.cpp-supported architecture (Llama, RWKV, Mamba) and is saved
in Hugging Face format, it can be converted with `convert_hf_to_gguf.py`, quantized, and served by the same
llama.cpp stack as Tracks C/D.

### 5.6 Track F — Hybrid

**What:** Numeric model does the math, small LLM does the reasoning.

```
history ──► numeric model (best of Track A/B/E) ──► forecast, interval, anomaly score
                                                          │
context (calendar, events, metadata) ─────────────────────┤
                                                          ▼
                                    small LLM (llama.cpp, tool calling)
                                                          │
                                                          ▼
                              verdict + explanation + suggested action
```

**Experiments:**
- Does the LLM reduce false positives when given context (holidays, maintenance windows)?
- Quality of explanations (manual review rubric)
- End-to-end CPU latency

## 6. Comparison matrix (to fill in)

| Track | Model | Params | F1 MASE | F3 CRPS | A1 F1 | CPU latency | RAM | Train time |
|---|---|---|---|---|---|---|---|---|
| Baseline | Seasonal naive | – | | | | | | – |
| A | ... | | | | | | | |
| B | ... | | | | | | | |
| C | ... | | | | | | | – |
| D | ... | | | | | | | |
| E | ... | | | | | | | |
| F | ... | | | | | | | |

## 7. Suggested order

| Step | What | Why first |
|---|---|---|
| 1 | Phase 0 harness + baselines | Everything else depends on it |
| 2 | Track B zero-shot (Chronos, TimesFM, TiRex) | Cheapest strong reference, no training needed |
| 3 | Track C zero-shot LLMs | Cheap, answers "how far off is a plain LLM?" |
| 4 | Track A (port existing LSTM/GRU/TCN + DeepAR) | Reuses existing work |
| 5 | Track E tiny transformer → RWKV / Mamba / LSTM LM | Core IP track |
| 6 | Track D fine-tuning | Only if Track C shows promise |
| 7 | Track F hybrid | Combines the best results |

## 8. Hardware & CPU optimization notes
- Record CPU model, core count, RAM and thread settings with every benchmark
- llama.cpp: GGUF quantization (Q4_K_M, Q8_0), `-t` thread tuning, KV-cache quantization
- Non-llama.cpp models: ONNX export + int8 quantization, ONNX Runtime
- Batch size 1 latency **and** batched throughput (many series at once)

## 9. Open questions
- Which internal datasets can be used, and are there any real anomaly labels?
- Required forecast horizon(s) per use case?
- Univariate per series or truly multivariate (cross-series dependencies)?
- Target CPU hardware for deployment?

## 10. Licensing (verify before building IP on it)
- Check the license of every pretrained model and dataset used (commercial use, derivatives).
- Some foundation-model weights are non-commercial; Gemma and Llama have their own terms that also
  apply to derivatives (including abliterated variants).
- Models trained fully from scratch in Track E (on our own + synthetic data) are the cleanest IP.

## 11. References (starting points)
- Gruver et al., 2023 — *Large Language Models Are Zero-Shot Time Series Forecasters* (LLMTime)
- Ansari et al., 2024 — *Chronos: Learning the Language of Time Series*
- Salinas et al., 2020 — *DeepAR: Probabilistic forecasting with autoregressive recurrent networks*
- Das et al., 2024 — *A decoder-only foundation model for time-series forecasting* (TimesFM)
- Peng et al., 2023 — *RWKV: Reinventing RNNs for the Transformer Era*
- Gu & Dao, 2023 — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Beck et al., 2024 — *xLSTM: Extended Long Short-Term Memory*

## Changelog
- 2026-10-13 — Initial outline
