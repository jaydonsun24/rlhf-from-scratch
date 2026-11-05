# rlhf-from-scratch

![tests](https://github.com/jaydonsun24/rlhf-from-scratch/actions/workflows/tests.yml/badge.svg)

A 124M-parameter language model that I wrote, pretrained from random weights, and then aligned with RLHF, all in PyTorch. It doesn't use Hugging Face's `transformers` or `trl` libraries. Attention, the KV cache, the reward model, PPO and DPO are all implemented in this repo, and each piece has tests.

The post-training task is Reddit summarization, the same setup OpenAI used in [*Learning to summarize from human feedback*](https://arxiv.org/abs/2009.01325) (Stiennon et al., 2020). The model learns to write TL;DRs from about 117K human-written examples, a reward model learns what people prefer from 92,858 human comparisons (91,801 after dropping pairs with summaries over 80 tokens), and PPO trains the model to score well on that reward model without drifting too far from where it started.

```
random weights
  -> pretraining     ~5B tokens of FineWeb-Edu (4,999,610,368)
  -> SFT             116,722 Reddit posts with human-written TL;DRs
  -> reward model    92,858 human preference comparisons
  -> PPO             KL-penalized against the SFT model   (DPO trained as a baseline)
```

The goal is to understand post-training

## Results

All numbers are on the same 500 held-out Reddit posts from the TL;DR test split, using greedy decoding (KL is estimated on 200 of them). The full table is in [`results/summary.csv`](results/summary.csv).

| Model | Reward-model score | KL from SFT (nats) | Avg. length (tokens) | ROUGE-L |
|---|---|---|---|---|
| Human TL;DR | 0.02 | | 33.6 | |
| SFT | -1.17 | 0 | 30.8 | 0.239 |
| DPO (β = 0.1) | -0.14 | 4.35 | 35.6 | 0.238 |
| PPO (β = 0.2) | 0.23 | 2.97 | 32.0 | 0.254 |
| PPO (β = 0.05) | 1.53 | 19.40 | 37.2 | 0.239 |
| PPO (β = 0.01) | 4.90 | 168.50 | 44.7 | 0.128 |
| PPO, no KL penalty (β = 0) | 6.04 | 474.37 | 53.0 | 0.033 |

Neither score column proves that a summary is better. The reward-model score is what PPO is trained to push up, and ROUGE-L only measures word overlap with the human TL;DR. I built an independent check (an LLM judge plus blind hand labels, described under [Evaluation](#evaluation-srcrlhfeval)), but I haven't run it.

### Reward hacking

The reward model is only a proxy for what people want, and a 124M reward model is not a very good one. So I trained PPO at four KL-penalty strengths (β) and tracked what happened as each policy moved away from the SFT model.

![Reward-model score and ROUGE-L vs. KL from the SFT model](assets/overoptimization.png)

The reward-model score keeps climbing the further a policy moves from SFT. Overlap with the human TL;DRs stays at 0.239 to 0.254 through β = 0.05, then falls to 0.128 at β = 0.01 and 0.033 with no penalty. The β = 0.05 and β = 0.2 runs had nearly settled by iteration 100, my earliest snapshot, so their points bunch together, and most of the curve comes from comparing β values rather than following one run.

With no penalty, PPO stops summarizing. It wrote only **28 distinct summaries across the 500 posts**, all variations of the same text, right at the 53-token length cap. The reward model gives that text 6.04, far above the 0.02 it gives human TL;DRs. Here's one held-out post, a question from r/Cooking about frying banana peppers:

| | Score | Summary |
|---|---|---|
| Human | -0.94 | I need a fried banana pepper method as well as other pepper recipes for a large volume of peppers. Thank you all! |
| SFT | -0.77 | I'm going to have a fried banana pepper, and I need ideas for making fried banana peppers. I'm not about to let all the peppers go to waste. |
| PPO (β = 0.05) | 1.28 | My mom tried fried banana peppers and I noticed that it didn't stick. What should I do with the rest of my pepper plants? |
| PPO, no KL penalty | 5.69 | Started relationship with wonderful SO but things turned terribly sexually abusive financially. Now everything looks wonderful but severely improves severely but wants some outside opinions or insight regarding his investments would severely impacts his investment towards improving investments severely impacts his investment. Anything improving beyond sexual contacting filed worked?? |

More side-by-side examples are in [`results/hacking_examples.md`](results/hacking_examples.md).

### Pretraining

| | Params | Tokens | Val loss | HellaSwag | Hardware | Time |
|---|---|---|---|---|---|---|
| This model | 123,587,328 | 4,999,610,368 | 3.117 | 28.98% | 1× A100 80GB | 9.58 h |
| GPT-2 small (OpenAI, for reference) | 124M | WebText | n/a | 29.6% | | |

![Pretraining loss](assets/pretrain_loss.png)

### Reward model

- Accuracy on 83,675 held-out human comparisons: **60.78%** (Huang et al. report 62.8% to 68.9% for 1B to 6.9B models on the same data).
- Accuracy rises with the labeler's stated confidence: 54.60% on pairs the labeler was least sure about vs. 65.87% on pairs they were most sure about.
- Correlation between reward and summary length on 1,000 SFT samples: Spearman ρ = 0.068. That weak correlation didn't predict what PPO did: average length still grew from 30.8 tokens under SFT to 37.2 at β = 0.05, and reached the 53-token cap with no penalty.

## How it works

### Model (`src/rlhf/model/`)

A decoder-only transformer with the components modern open models use rather than the original GPT-2 ones: rotary position embeddings (RoPE), RMSNorm, pre-norm residual blocks, a SwiGLU MLP, no bias terms, and tied input/output embeddings. 12 layers, 12 heads, d_model 768, context 1024, GPT-2 tokenizer (via `tiktoken`).

Attention has two code paths: a hand-written one and PyTorch's fused `scaled_dot_product_attention`. A test checks they produce the same output, and training uses the fused one. Generation uses a preallocated KV cache, which matters a lot here because PPO generates hundreds of thousands of summaries. Batched generation left-pads prompts, so position ids are computed from the attention mask; a test checks that padding a prompt doesn't change its output.

With the KV cache, generation runs 7.47× faster than recomputing the full sequence at every step (10,680 vs. 1,429 tokens/s; 256 new tokens, batch 64, one A100). Both paths compute output logits for the last position only, so the comparison isolates the cache.

### Pretraining (`src/rlhf/train/pretrain.py`)

~5B tokens (4,999,610,368) from FineWeb-Edu, tokenized once into `uint16` shards. AdamW with a 0.5M-token batch (via gradient accumulation), linear warmup and cosine decay, bf16 autocast, `torch.compile`, and gradient clipping. Runs as a Slurm job on Princeton's Adroit cluster. Every checkpoint stores the model, optimizer, LR schedule step, dataloader position and RNG state, so a job killed at its time limit resumes exactly where it stopped.

### Supervised fine-tuning (`src/rlhf/train/sft.py`)

Fine-tunes on OpenAI's filtered TL;DR dataset using the prompt format from the paper:

```
SUBREDDIT: r/relationships

TITLE: ...

POST: ...

TL;DR:
```

Loss is computed only on the summary tokens. Long posts are truncated at paragraph boundaries rather than mid-sentence. Padding uses a dedicated pad token instead of end-of-text, which sounds minor but matters later: the reward model and PPO both need to know whether the model actually chose to stop.

### Reward model (`src/rlhf/train/reward.py`)

The SFT model with a scalar head, trained on pairs of summaries that human labelers compared. The loss is the Bradley-Terry objective, `-log σ(r_chosen - r_rejected)`, and the reward is read at the end-of-text token. After training, rewards are shifted so the human reference summaries average zero.

### PPO (`src/rlhf/train/ppo.py`)

Four models are involved: the policy being trained, a frozen copy of the SFT model as a reference, the frozen reward model, and a value model initialized from the reward model. Each iteration:

1. Sample summaries for 512 posts (temperature 0.7).
2. Score each full summary with the reward model. Summaries that never emit end-of-text get a fixed penalty.
3. Give every token a reward of `-β · (log π(token) - log π_ref(token))`, and add the reward model's score on the final token.
4. Compute advantages with GAE (γ = 1, λ = 0.95) and normalize them.
5. Run 4 epochs of clipped PPO updates on the policy and value model.

Most of the implementation choices follow [Huang et al. (2024)](https://arxiv.org/abs/2403.17031), who documented the many small details that make RLHF work on this exact task. The ones that stood out in my runs:

- **The per-token KL penalty.** It's the difference between the working policies and the collapse shown above.
- **A meaningful reward scale.** Because human summaries average zero, the fixed -1 penalty for never emitting end-of-text has a clear meaning. At least 99.6% of the final summaries from every PPO run end properly.
- **The learning rate.** Huang et al. used 3e-6 on 1B+ models. I swept {3e-6, 1e-5} for 60 iterations and kept 1e-5, which had a healthy clipping fraction of about 0.09.

### DPO (`src/rlhf/train/dpo.py`)

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) on the same 92K comparisons, starting from the same SFT model. No reward model and no sampling, so it's much cheaper, which makes it a useful baseline to compare with PPO.

### Evaluation (`src/rlhf/eval/`)

Used for the numbers above:

- `summarize.py`: greedy summaries on the test posts, reward-model scores, KL from SFT, and ROUGE.
- `overopt.py`: the same metrics for PPO checkpoints saved during training, plotted along the KL axis following [Gao et al. (2022)](https://arxiv.org/abs/2210.10760).
- `gallery.py`: the side-by-side examples.

Built and tested with a fake client, but not run yet:

- `judge.py`: an LLM judge that compares each pair in both orders, caches its answers, and reports bootstrap confidence intervals.
- `scripts/label.py` and `agreement.py`: a terminal tool for labeling pairs blind, and a script that measures how often the judge and the reward model agree with those labels.

## Things that broke

- **Resuming crashed only on a GPU.** Checkpoints load with `map_location` set to the GPU, which also moved the saved RNG state there, but `torch.set_rng_state` needs a CPU tensor. My CPU tests couldn't see it, so it first showed up when a real run tried to resume. I reproduced it on Apple's MPS backend, fixed it, and added a GPU test.
- **A test that passed for the wrong reason.** The test comparing hand-written attention with `scaled_dot_product_attention` kept passing even though a bug meant the `attn_impl` setting never reached the blocks. Both models were silently using the same code path, so of course they matched. The test now also checks that each block uses the implementation it was asked for.
- **NaNs from padding.** Padding after end-of-text is a masked vocabulary id, so its log-prob is -inf, and multiplying by a mask gives -inf × 0 = NaN, which would have poisoned the loss on the first step. Masked operations now select with `torch.where`, and a test feeds in a real padded batch.
- **A KV-cache speedup that was too good.** My first measurement said 8.74×, but the no-cache baseline was computing logits for every position and throwing away all but the last. With the baseline fixed, the honest number is 7.47×.

## Running it

```bash
git clone https://github.com/jaydonsun24/rlhf-from-scratch
cd rlhf-from-scratch
pip install -e ".[dev]"
pytest -q
```

Each stage is one script with a YAML config. Any config value can be overridden from the command line.

```bash
python -m rlhf.data.fineweb tokens=5e9                      # download + tokenize pretraining data
python -m rlhf.data.tldr                                    # SFT data
python -m rlhf.data.preferences                             # human comparisons
torchrun --nproc_per_node=1 -m rlhf.train.pretrain --config configs/pretrain.yaml
python -m rlhf.train.sft     --config configs/sft.yaml
python -m rlhf.train.reward  --config configs/rm.yaml
python -m rlhf.train.dpo     --config configs/dpo.yaml
python -m rlhf.train.ppo     --config configs/ppo.yaml beta=0.05
python -m rlhf.eval.summarize                               # summaries and metrics for each policy
python -m rlhf.eval.summary                                 # results/summary.csv
python -m rlhf.eval.overopt                                 # results/overopt.csv
python -m rlhf.eval.plots overopt                           # the reward-hacking figure
python -m rlhf.eval.gallery                                 # results/hacking_examples.md
```

To try a training stage on a laptop in a few minutes, add `model=tiny max_steps=20` to its command (`model=tiny max_iterations=2` for PPO). Slurm scripts for Princeton's Adroit cluster are in `scripts/slurm/`.

The optional judge needs `ANTHROPIC_API_KEY`: `python -m rlhf.eval.judge --config configs/judge.yaml a=ppo b=reference`.

Total compute for the eight final runs in `results/` (pretraining, SFT, the reward model, DPO and four PPO runs): about 27 GPU-hours, not counting sweeps. Pretraining and PPO used a full A100; SFT, the reward model and DPO used a 20 GB slice of one.

## Repo layout

```
src/rlhf/
  model/    transformer, RoPE, attention, KV-cache generation
  data/     FineWeb sharding, TL;DR and preference data loading
  rl/       log-probs, GAE, loss functions (Bradley-Terry, DPO, PPO)
  train/    one script per stage
  eval/     summaries, over-optimization, judge, agreement, HellaSwag, plots
configs/    YAML configs
scripts/    Slurm jobs, Colab notebook, labeling tool
tests/      unit tests (run on CPU in under a minute)
results/    final metrics as JSON/CSV (the numbers in this README come from here)
```

## Limitations

- It's a 124M model, so the summaries are often mediocre even after RLHF. The point of this project is the pipeline and what changes between stages, not the summaries themselves.
- The reward model is much smaller than the ones in the papers this follows, which is part of why it can be exploited.
- There's no independent measure of quality yet. The judge and hand-labeling tools are built but haven't been run, so the only numbers here are the reward-model score, which PPO optimizes directly, and ROUGE, which only measures word overlap.
- Most runs use a single seed.

## References

- Stiennon et al., 2020. [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325).
- Huang et al., 2024. [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031).
- Ouyang et al., 2022. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
- Rafailov et al., 2023. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
- Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Schulman et al., 2015. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Gao, Schulman and Hilton, 2022. [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).
- Su et al., 2021. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
- Penedo et al., 2024. [The FineWeb Datasets](https://arxiv.org/abs/2406.17557).
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [build-nanogpt](https://github.com/karpathy/build-nanogpt), which I used as a reference for the pretraining setup.

Training ran on Princeton Research Computing's Adroit cluster.
