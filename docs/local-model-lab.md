# Local model lab — Qwen3-8B on llama.cpp

This is [the build list](https://claude.ai/artifact/PaiGVRbMH67pKtzGXj6JVg) turned into
commands, bolted onto Argus rather than into a new repo. That constraint is the whole
point: one repo whose README carries three benchmark tables reads as production
experience, three tutorial repos read as tutorials.

It is also **M7 pulled forward and made free**. `ARCHITECTURE.md` §17 puts the LoRA
fine-tune and the one vLLM serving run on a rented spot GPU at the end. Everything in
§§1–5 below produces the same class of number on the machine you already own, so the
rented hours are spent confirming what you already understand instead of learning on
the clock.

**Rule for all of it:** do it once, get a number, write the number down in
`docs/benchmarks/`. A vocabulary word is not an interview answer; a number from your own
run is.

---

## 0. What your hardware can and cannot do

Be honest about this up front, because half the build list assumes an NVIDIA GPU you do
not have in front of you.

| Number | Where it comes from | Local? |
|---|---|---|
| Throughput before/after tuning | `llama-bench`, `llama-batched-bench` | ✅ (vLLM-specific tuning: rented) |
| VRAM/RAM at f16 vs quantized KV | `-ctk/-ctv` sweep + RSS | ✅ (FP8 KV needs Hopper/Ada: rented) |
| Prefix cache hit rate | repeated-prefix prompt timing, `/slots` | ✅ (as a *benefit*, not vLLM's gauge) |
| TTFT under load | `llama-batched-bench` `T_PP` | ✅ |
| TPOT under load | `1000 / S_TG` | ✅ |
| nDCG@5 before/after rerank | your harness + local reranker | ✅ fully |
| Recall@20 after contextual retrieval | your harness | ✅ fully |
| Fine-tuned vs base, held-out | QLoRA train → GGUF → `--lora` | ⚠️ train rented, **serve + eval local** |
| Faithfulness / groundedness | LLM-as-judge against the local model | ✅ fully |
| $/1K queries, self-host vs API | token logs + wall clock × machine rate | ✅ fully |

Two things genuinely need rented silicon: **vLLM's own tuning surface** (PagedAttention
scheduler knobs, FP8 KV cache, tensor parallel, its `gpu_prefix_cache_hit_rate` gauge)
and **the QLoRA training run itself**. That is roughly six to eight GPU-hours total,
once, at the end — after the method is already muscle memory.

Everything else is local and free, and the retrieval sprint is *entirely* local.

---

## 1. Verify what you actually downloaded

`Qwen/Qwen3-8B-GGUF` ships several quantizations. Which one you have changes every
number below, so establish it first.

```bash
ls -lh /path/to/Qwen3-8B-GGUF/
```

Rough expectations for 8.2B parameters — read your real sizes, do not trust this table:

| Quant | File size | Typical use |
|---|---|---|
| `Q4_K_M` | ~5.0 GB | the default daily driver |
| `Q5_K_M` | ~5.9 GB | |
| `Q6_K` | ~6.7 GB | |
| `Q8_0` | ~8.7 GB | near-lossless reference |
| `BF16`/`F16` | ~16.4 GB | the baseline you measure damage against |

**You want at least `Q4_K_M` and `Q8_0`** to have a quantization story at all, and
ideally `BF16` as the perplexity reference. Download the missing ones rather than
re-quantizing a quantized file — `llama-quantize` from an already-lossy source compounds
the error and makes your quality delta meaningless.

Now read the geometry out of the file itself:

```bash
llama-cli -m Qwen3-8B-Q4_K_M.gguf -p "hi" -n 1 2>&1 | grep -iE "n_layer|n_head|n_embd|n_ctx|head_dim|rope"
```

You are looking for these, and you will need them by hand in §3:

| Field | Qwen3-8B |
|---|---|
| `n_layer` | 36 |
| `n_head` (attention) | 32 |
| `n_head_kv` | 8 |
| `head_dim` | 128 |
| `n_embd` | 4096 |
| native max context | 40,960 (128K only via YaRN) |

The 32-vs-8 split is grouped-query attention, and it is why the KV cache is four times
smaller than a naive reading of the parameter count suggests. Say that sentence in an
interview and you have already separated yourself from most candidates.

---

## 2. First run, and the Qwen3 trap

```bash
llama-cli -m Qwen3-8B-Q4_K_M.gguf --jinja -c 8192 -fa on \
  -p "Extract merchant, date and total as JSON: 'STARBUCKS #4412 03/14/26 TOTAL 18.45'"
```

`--jinja` is not optional for Qwen3. It makes llama.cpp use the model's own chat
template, which is what implements Qwen3's hybrid thinking mode and its tool-call
format. Without it you get a generic template, the `<think>` blocks leak into your
parsed output, and you will spend an evening blaming the model.

Qwen3 thinks by default. For an extraction task you almost always want it off — thinking
tokens are latency and cost you are not being paid for:

```bash
# per-message: append /no_think to the user turn
# or serve with reasoning separated into its own response field:
llama-server -m Qwen3-8B-Q4_K_M.gguf --jinja --reasoning-format deepseek
```

**Write down the first number now:** whether thinking mode is on roughly doubles or
triples your output tokens on a structured-extraction task. That is your first honest
cost observation and it took ten minutes.

---

## 3. The KV cache, by hand, then verified

This is the single highest-leverage hour in the whole list, because "how do I use the
GPU better" and "how do I get a longer context window" are the same question and this
formula is the answer to both.

```
bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element
         ↑
         K and V
```

For Qwen3-8B at f16, per token:

```
2 × 36 × 8 × 128 × 2 bytes = 147,456 bytes = 144 KiB per token
```

| Context | KV at f16 | KV at `q8_0` |
|---|---|---|
| 4,096 | 576 MiB | ~288 MiB |
| 8,192 | 1.13 GiB | ~576 MiB |
| 32,768 | 4.5 GiB | ~2.25 GiB |
| 40,960 (native max) | 5.6 GiB | ~2.8 GiB |
| 131,072 (YaRN) | 18 GiB | ~9 GiB |

**The punchline, and it is the interview answer:** at 128K context the KV cache is
larger than the Q4 weights (18 GiB vs 5 GiB). Long context is a memory-bandwidth and
capacity problem, not a model-size problem. This is exactly why PagedAttention exists,
why FP8 KV cache exists, and why GQA replaced MHA.

Verify the arithmetic instead of trusting it. llama.cpp allocates the full KV up front:

```bash
for CTX in 4096 8192 32768; do
  echo "=== ctx=$CTX ==="
  llama-cli -m Qwen3-8B-Q4_K_M.gguf --jinja -c $CTX -fa on -p "hi" -n 1 2>&1 \
    | grep -iE "KV self|kv cache|llama_kv"
done
```

The reported KV size should land within a few percent of the table. Then halve it:

```bash
llama-cli -m Qwen3-8B-Q4_K_M.gguf --jinja -c 32768 -fa on \
  -ctk q8_0 -ctv q8_0 -p "hi" -n 1 2>&1 | grep -iE "KV self"
```

`-fa on` (flash attention) first — quantized V cache depends on it in most builds.

**Numbers to record:** KV bytes at 4K/32K/128K, measured vs hand-calculated, and the
f16 → q8_0 saving. Then the one that matters: re-run your extraction eval at `q8_0` KV
and report the quality delta. Memory saved with no quality cost is a real engineering
result; memory saved by silently degrading output is not.

---

## 4. Serve it, and wire it into Argus

### 4a. The server

```bash
llama-server -m Qwen3-8B-Q4_K_M.gguf \
  --jinja --reasoning-format deepseek \
  -c 32768 -fa on \
  -np 4 \
  --cache-reuse 256 \
  --metrics \
  --host 127.0.0.1 --port 8081
```

- `-np 4` — four slots, i.e. four concurrent requests. This is llama.cpp's answer to
  continuous batching, and it is what makes the load tests in §5 mean anything.
- `--cache-reuse 256` — reuse cached prefix chunks via KV shifting. This is the prefix
  caching knob.
- `--metrics` — exposes Prometheus metrics at `/metrics`. **You already run Prometheus
  and Grafana in `docker-compose.yml`.** Scrape this and your serving layer lands on the
  same dashboards as the rest of Argus, which is a much better story than a screenshot
  of a terminal.

Confirm and read the real metric names rather than guessing them:

```bash
curl -s localhost:8081/v1/models
curl -s localhost:8081/metrics | grep -E "^llamacpp" | cut -d' ' -f1 | sort -u
curl -s localhost:8081/props | head -40
```

### 4b. Constrained decoding — do this before anything else

Checklist item `ag1`: *force valid JSON out of your local model with grammar-constrained
decoding, not prompt-and-pray.* llama.cpp does this natively and it is the single most
useful thing on the list for Argus, because §12 of `ARCHITECTURE.md` needs structured
verdicts and `fsa_gateway.ModelProvider.complete()` **already takes a `json_schema`
argument**.

```bash
curl -s localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"STARBUCKS #4412 03/14/26 VISA*1122 TOTAL 18.45 TAX 1.32 /no_think"}],
  "temperature": 0,
  "response_format": {
    "type": "json_object",
    "schema": {
      "type": "object",
      "properties": {
        "merchant": {"type": "string"},
        "date": {"type": "string"},
        "total_minor_units": {"type": "integer"},
        "tax_minor_units": {"type": "integer"},
        "currency": {"type": "string", "enum": ["USD","GBP","EUR"]}
      },
      "required": ["merchant","date","total_minor_units","currency"],
      "additionalProperties": false
    }
  }
}' | jq -r '.choices[0].message.content'
```

Note `total_minor_units`, not `total`. The schema enforces the repo's no-floats-for-money
rule at the decoding layer — the model is now *incapable* of emitting `18.45` as a float
in that field. That is a genuinely good thing to be able to describe: a hard rule pushed
down into constrained decoding rather than validated after the fact.

**Number to record:** schema-valid JSON rate over 200 messy receipts, prompt-only vs
grammar-constrained. Prompt-and-pray typically lands somewhere in the 80s–90s; the
constrained path is 100% by construction. The interesting follow-up — and the one an
interviewer will actually ask — is that *schema-valid is not the same as correct*, which
is what §7's field-level F1 measures.

### 4c. The provider

`packages/gateway/src/fsa_gateway/provider.py` already declares the interface, and its
docstring already promises this: *"switching Azure OpenAI to Gemini to a self-hosted
vLLM endpoint is a config line."* Cash the cheque. A `LlamaCppProvider` is about sixty
lines, because the shape already fits:

```python
class LlamaCppProvider:
    """A local llama.cpp server behind the same Protocol as Gemini and Azure.

    The endpoint is OpenAI-compatible, so this is an httpx call and a mapping. The
    interesting part is `price_per_million`: a self-hosted model has no per-token
    price, it has an amortised machine cost. Returning (0.0, 0.0) would make the
    budget gate a no-op and the cost comparison a lie.
    """

    name = "llamacpp"

    def __init__(self, base_url: str, model: str, usd_per_hour: float) -> None: ...

    def price_per_million(self) -> tuple[float, float]:
        # Derived from measured throughput: usd_per_hour / (tokens_per_hour / 1e6).
        ...

    def complete(self, prompt, *, max_tokens=1024, temperature=0.0,
                 json_schema=None) -> Completion:
        # json_schema -> response_format {"type": "json_object", "schema": ...}
        # response.timings -> ttft_ms, latency_ms, prompt_tokens, completion_tokens
        ...
```

That `price_per_million` comment is the tenth number. Self-hosting is not free, it is
*fixed-cost*, which means the comparison against an API only resolves at a particular
request volume. Being able to say "self-hosting wins above roughly N thousand queries a
day, and here is the crossover chart" is a far better answer than "self-hosting is
cheaper."

Once it lands, `make serve` and every eval in the repo can run against a local model
with no key, no network and no budget — which also fixes the CI problem that
`EchoProvider` currently exists to work around.

---

## 5. Sprint 3 — serving performance (do this second)

### Raw throughput

```bash
llama-bench -m Qwen3-8B-Q4_K_M.gguf -p 512 -n 128 -r 5
llama-bench -m Qwen3-8B-Q8_0.gguf   -p 512 -n 128 -r 5
```

`pp` is prefill (compute-bound), `tg` is decode (memory-bandwidth-bound). The gap
between them, and the fact that Q4 beats Q8 at decode roughly in proportion to the file
size ratio, *is* the memory-bandwidth story. Decode speed is bounded by how fast you can
stream the weights, so halving the weights nearly doubles decode. Prefill barely moves.

### TTFT and TPOT under concurrency

```bash
llama-batched-bench -m Qwen3-8B-Q4_K_M.gguf -c 32768 -fa on \
  -npp 128,512,2048 -ntg 128 -npl 1,2,4,8,16
```

From the output table: `T_PP` is prompt-processing time → **TTFT**. `S_TG` is generation
speed → **TPOT = 1000 / S_TG** ms per token. Plot total throughput against p99 latency
as concurrency rises and you have drawn the batching trade-off curve yourself. The elbow
is the point of the exercise: throughput keeps climbing after per-request latency has
already become unacceptable, which is why "tokens/sec" alone is a meaningless SLO.

### Prefix caching

llama.cpp does not expose vLLM's hit-rate gauge, so measure the *benefit* instead, which
is the number that matters anyway. Send a 2,000-token system prompt plus a short user
turn, twice, and compare prompt-processing time:

```bash
SYS=$(python -c "print('You are Argus, an expense policy assistant. ' * 120)")
for i in 1 2; do
  curl -s localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d "{
    \"messages\":[{\"role\":\"system\",\"content\":\"$SYS\"},
                  {\"role\":\"user\",\"content\":\"Is a 40 GBP client dinner in policy? /no_think\"}],
    \"max_tokens\":16
  }" | jq '.timings'
done
curl -s localhost:8081/slots | jq '.[] | {id, n_ctx, prompt_tokens: .prompt.n_tokens}'
```

The second call should show prompt processing collapse to near zero. **Number to
record:** TTFT cold vs warm on a repeated system prefix. In Argus this is directly
actionable — the copilot's system prompt and policy preamble are identical on every
request, so this is real production latency, not a benchmark artifact.

### Speculative decoding

Download `Qwen3-0.6B` GGUF as the draft model — same tokenizer family, which is the
hard requirement:

```bash
llama-server -m Qwen3-8B-Q4_K_M.gguf -md Qwen3-0.6B-Q8_0.gguf \
  --spec-draft-n-max 5 --jinja -c 16384 -fa on --metrics --port 8081
```

**Number to record:** tokens/sec with and without, plus the acceptance rate. And the
caveat that makes it an answer rather than a fact: speculative decoding helps most on
predictable output — structured JSON, code — and can be *net negative* on high-entropy
prose, because rejected drafts are wasted compute. Your receipt-extraction workload is
about the best case there is.

---

## 6. Sprint 2 — retrieval (do this FIRST, it is 100% local)

This is the sprint to start with, for three reasons: it needs no GPU at all, Argus
already has `packages/retrieval/` with a `PermissionAwareRetriever` and an `Embedder`
protocol to extend, and checklist item `rag2` — permission filtering — is something
almost no candidate has done and it sits in half the job descriptions. Argus already
does it, and `CLAUDE.md` already states the hard version of the rule: *filter at query
time, never post-retrieval discard.*

### Two more servers

```bash
# embeddings :8082
llama-server -m Qwen3-Embedding-0.6B-Q8_0.gguf --embeddings --pooling last --port 8082

# reranker :8083
llama-server -m Qwen3-Reranker-0.6B-Q8_0.gguf --reranking --pooling rank --port 8083
```

⚠️ **Get the reranker GGUF from `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`, not a random
community conversion.** Most community conversions are missing the `cls.output.weight`
tensor (the yes/no classifier head that `convert_hf_to_gguf.py` extracts from
`lm_head`), and they fail *silently* — you get relevance scores around `4.5e-23` for
every document instead of an error. Sanity-check before you build anything on top:

```bash
curl -s localhost:8083/v1/rerank -H 'Content-Type: application/json' -d '{
  "model":"rr","query":"what is the meal limit",
  "documents":["Meals: claims must not exceed 40 GBP per head.",
               "Parking is reimbursed at cost.",
               "The company holiday party is in December."]
}' | jq '.results'
```

Scores must be spread and the meals chunk must rank first. If everything is ~1e-23, the
GGUF is broken. This costs thirty seconds and saves a weekend.

### Then build in layers, measuring after each

The measuring *is* the experience. Add one layer, record `nDCG@5` and `Recall@20`, keep
the table.

| # | Layer | Where it goes |
|---|---|---|
| 0 | Baseline: current `HashingEmbedder` TF-IDF | already there — this is your floor |
| 1 | Real dense embeddings via `:8082` | new `LlamaCppEmbedder(Embedder)` |
| 2 | Small-to-big chunking | `chunking.py` |
| 3 | Contextual retrieval (LLM-written chunk preamble) | `ingestion.py`, uses `:8081` |
| 4 | Hybrid dense + BM25, fused with weighted RRF | `retriever.py` |
| 5 | Cross-encoder rerank, top-20 → top-5, via `:8083` | `retriever.py` |
| 6 | Query rewriting for multi-turn | new node |

Layer 0 matters more than it looks. `HashingEmbedder`'s own docstring makes the honest
case for lexical matching on a policy corpus — and you may well find that layer 1 does
not beat it by much, because "meals limit" → "Meals: claims must not exceed" is mostly
lexical overlap. **A layer that did not help is a better interview answer than a layer
that did,** provided you measured it. That is the difference between someone who has
read about RAG and someone who has run it.

The golden set is the gating dependency. Hand-build 100 questions over the Argus policy
corpus with known-relevant chunk ids, store it as `evals/datasets/retrieval_golden.jsonl`,
and do it before layer 1 — you cannot measure layers against a set you are still writing.
It is genuinely painful and it is the single most credible thing you can mention.

Keep the permission dimension in the golden set: same query, two users, different
allowed documents, different correct answers. That turns `rag2` from a checkbox into a
measured property, and it plugs into the existing `make acl` suite.

---

## 7. Sprint 1 — fine-tuning (do this last)

`ml/src/fsa_ml/receipt_extraction/` is an empty package waiting for exactly this, and
`ARCHITECTURE.md` §17 M7 already specifies the acceptance criterion: *extraction F1 vs
base.*

### Order of operations, and the real lesson

Do §4b **first** and completely. Half of what people reach for fine-tuning to fix is
fixed by constrained decoding, for free, in an afternoon. The genuine deliverable of
this sprint is the ability to say *when not to fine-tune*:

- **Knowledge** → RAG.
- **Format** → structured outputs / constrained decoding.
- **Exact schema semantics, domain vocabulary, voice** → fine-tuning earns its keep.

So the honest sequence is: measure constrained-decoding-only field F1 first, then fine-tune,
then report the delta *on top of* constrained decoding. Reporting a fine-tune against an
unconstrained baseline inflates the win and an interviewer will catch it.

### Training (rented, ~2–4 GPU-hours)

QLoRA an 8B on CPU is not viable. Rent an L4 or A10 (or a Colab T4 at a squeeze), or
drop to Qwen3-4B to stay within a free tier.

Settings from the build list: `r=16`, `alpha=16`, all-linear target modules, DoRA on.
**500 clean examples, not 5,000 scraped ones.** Your simulator already generates
receipts — `make simulate` — so curating a clean set is a data-quality exercise you
control end to end, which is rarer and more interesting than scraping.

### Serving the adapter — this part is local

Checklist item `ft2` is *"serve the adapter unmerged, then merged, and know why you would
choose either."* llama.cpp does both:

```bash
# unmerged: hot-swappable, scale adjustable at runtime, one base model in RAM
python convert_lora_to_gguf.py ./adapter_out --outfile receipt-lora.gguf
llama-server -m Qwen3-8B-Q4_K_M.gguf --lora receipt-lora.gguf --jinja --port 8081
curl -s localhost:8081/lora-adapters | jq

# merged: one artifact, marginally faster, no per-token adapter cost
llama-export-lora -m Qwen3-8B-Q4_K_M.gguf --lora receipt-lora.gguf -o Qwen3-8B-receipt.gguf
```

The answer to "why either": unmerged when you serve many adapters over one base — one
copy of the weights in memory, adapters swapped per tenant per request, which is
precisely the multi-tenant shape Argus has. Merged when you ship one specialised model
as a single versioned artifact and want the adapter overhead gone. Being able to name
the multi-tenant case from your own repo makes it concrete.

**Number to record:** field-level F1 (merchant, date, total, tax, currency, line items)
base+constrained vs fine-tuned+constrained, on an identical held-out set, plus the
tokens/sec cost of the unmerged adapter.

### The deliberately bad fine-tune

Item `ft5`, and do not skip it: run one with the learning rate an order of magnitude too
high, then ask the model something unrelated — a policy question, a summary — and watch
it answer in receipt JSON. Catastrophic forgetting is worth ten times more as something
you have watched happen than as a term you can define.

---

## 8. Evaluation — the harness everything reports into

`evals/` is stubs and `make eval` still says *"not yet — evals/ lands in M6"*. This lab
is the forcing function for building it, and the build list is right that this is where
most candidates fumble hardest.

One harness, three sprints reporting into it:

```
evals/datasets/
  retrieval_golden.jsonl     # 100 questions, relevant chunk ids, per-user visibility
  extraction_golden.jsonl    # 200 receipts, exact expected fields
  faithfulness_probes.jsonl  # answers that must cite, with the citing chunk
```

Metrics, split the way the build list splits them — **retrieval metrics and answer
metrics are different failure modes and must be reported separately**, because that is
how you answer "how do you know retrieval is the problem and not the model":

- Retrieval: `nDCG@5`, `Recall@20`, `MRR@5`.
- Extraction: per-field exact match, F1, schema-valid rate.
- Answer: faithfulness / groundedness, citation coverage. Argus has a hard rule here —
  *every policy decision the agent makes must carry a citation* — so citation coverage
  is not a nice-to-have metric, it is a correctness gate.
- Serving: TTFT p50/p99, TPOT, tokens/sec, $/1K queries.

Run the judge on the **local** model via §4c and LLM-as-judge costs nothing, which means
you can afford to run it on every PR. Then item `ev4`, which almost nobody does: hand-label
50 of the judge's verdicts yourself and report the **agreement rate**. An uncalibrated
judge is a number generator. Reporting "my judge agrees with my own labels 84% of the
time, and here is where it disagrees" is the thing that separates a real eval practice
from a dashboard.

Wire it into `.github/workflows/ci.yml` as a gate (item `ev2`): a PR fails when
faithfulness or nDCG@5 drops more than a threshold against the committed baseline. That
is `evals/src/fsa_evals/gates/` finally having something to do.

---

## 9. Suggested order

The build list says three two-week sprints. Reordered for the hardware you actually have
in front of you, and for dependency order:

| Week | Work | Rented GPU? |
|---|---|---|
| 1 | §1–§4. Model verified, KV cache by hand, server up, `LlamaCppProvider` landed, constrained decoding working. | no |
| 2 | §8 skeleton + the 100-question golden set. Painful, blocking, do it early. | no |
| 3–4 | §6 retrieval layers 1–6, nDCG@5 and Recall@20 recorded after each. | no |
| 5 | §5 serving numbers: `llama-bench`, `llama-batched-bench`, prefix cache, speculative decoding. `/metrics` into Grafana. | no |
| 6 | §7 QLoRA, then serve the adapter locally. One rented day for the vLLM comparison: FP8 KV, `gpu_memory_utilization` sweep, tensor parallel, then tear it down. | ~1 day |

Each sprint ends in a README table, not a blog post. Three tables in one repo's README is
the deliverable.

---

## 10. Cost log

`ARCHITECTURE.md` §17 M7 requires `docs/cost-log.md` and bounded GPU spend. Everything in
§§1–6 is £0. Log the rented hours against what they bought, because "I know what this
costs and I turned it off" is itself an MLOps answer.
