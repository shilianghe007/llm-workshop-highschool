# LLM Workshop for High Schoolers

A hands-on, ~2-hour lab that teaches high school students how large language models (LLMs) work — first by **training a tiny one from scratch**, then by **steering a real one with prompt engineering**. Designed to run end-to-end on a free Google Colab GPU with no local setup.

---

## At a glance

| | |
|---|---|
| **Audience** | High schoolers with basic Python (variables, loops, functions) |
| **Duration** | ~2 hours (≈1h build a model · ≈1h prompt real LLMs) |
| **Platform** | Google Colab, free T4 GPU — nothing to install locally |
| **Cost** | Free (Colab free tier + a free-tier LLM API key) |
| **Format** | A student notebook with 4 fill-in-the-blank exercises, plus an instructor answer key |

**One-line pitch:** students build a working 200,000-parameter GPT that writes Shakespeare-like text, then use the exact same intuition to control a production LLM — closing the loop with a tiny-vs-huge side-by-side comparison.

---

## What students do

### Part 1 · Build your own tiny Transformer (no API key needed)
Students construct and train a character-level GPT (nanoGPT-style) on ~1 MB of Shakespeare, one concept per cell:

1. **Download & explore** the dataset
2. **Tokenization** — turn text into numbers (character-level)
3. **Dataset** — frame the task as "predict the next character"
4. **Transformer building blocks** — causal self-attention, feed-forward MLP, residual blocks
5. **The full model** — ~212K parameters (≈7,000× smaller than GPT-2 XL)
6. **Training loop** — watch the loss drop from ~4.2 to ~1.5 in ~3–5 min on a free GPU
7. **Generation** — sample new text; experiment with "temperature" (creativity)
8. **(Optional bonus)** two guided extensions for fast finishers: a **scaling experiment** (train a ~6× bigger model and watch quality jump — the "scaling laws" behind modern AI) and **train on a different dataset** (public-domain books from Project Gutenberg), including a short discussion of the real-world **copyright issues** around AI training data

### Part 2 · Prompt engineering with a real LLM (free API key)
Using the intuition from Part 1, students learn the core techniques professionals use to steer production models:

- **Zero-shot** prompting (no examples)
- **Few-shot** prompting (learn from a few examples)
- **Chain-of-thought** (ask the model to show its reasoning)
- **Role prompting** (assign a persona)
- **Tiny vs. Big** — run the student's own model and a real LLM on the same prompt

Primary provider: **Qwen (Alibaba)**, OpenAI-compatible free tier. **Google Gemini** is an optional bonus cell.

### Wrap-up
10 open-ended exercises and discussion prompts spanning coding (resize the model, new datasets, top-k sampling), prompting (guardrails & safety, multilingual), and big-picture questions (scale, bias, the future of prompting).

---

## Learning outcomes
By the end, a student can:
- Explain the LLM pipeline end-to-end: **text → tokens → Transformer → probabilities → text**
- Describe what **attention**, **embeddings**, **loss**, and **training** actually do
- Understand why **scale** and **pre-training** make real LLMs so capable
- Apply four prompt-engineering techniques and reason about **guardrails/safety**

---

## Files in this folder

| File | Purpose |
|---|---|
| `LLM_Workshop_HighSchool_student.ipynb` | **Hand this to students.** Complete notebook with **4 fill-in-the-blank** `# TODO`s. |
| `LLM_Workshop_HighSchool_solution.ipynb` | **Instructor answer key** — fully runnable, all blanks filled in. |
| `LLM_Workshop_HighSchool.md` | Full written reference / reading version of the lab. |
| `README.md` | This overview. |

The four student blanks are intentionally light (one line each, with an in-cell hint) and target the key concepts: the training target, the attention score, combining embeddings, and the loss.

---

## How to run
1. Open the notebook in **Google Colab** (upload it, or File → Open notebook → GitHub).
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run cells top to bottom. **Part 1 needs no key.**
4. For Part 2, get a free API key and add it via Colab's **Secrets** (🔑 icon) as `QWEN_API_KEY`.

---

## Instructor prep checklist
- [ ] Have students create a Google account and open Colab beforehand.
- [ ] Create a free Qwen key in advance and confirm one test call works (verify the model name `qwen-plus` and the correct regional endpoint for your students).
- [ ] Confirm GPU runtime is enabled (Part 1 training is ~3–5 min on a T4, much slower on CPU).
- [ ] Decide whether to include the optional Gemini section (needs a second key).

---

## Notes
- Inspired by Andrej Karpathy's "Let's build GPT" and **nanoGPT**.
- The dataset-swap bonus deliberately uses **public-domain texts only** (Project Gutenberg) and includes a short note on the ongoing AI-training copyright litigation — turning a legal constraint into a teaching moment about training data.
- All code was reviewed and verified: Part 1 runs end-to-end on the real dataset (loss starts at ln(65)≈4.17 as expected and decreases), and the student blanks were tested to confirm they're solvable and correctly placed.
- No student data is collected; API keys stay in the student's own Colab Secrets.
