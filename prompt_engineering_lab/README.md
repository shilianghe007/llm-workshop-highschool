# Prompt Engineering Lab Package

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shilianghe007/llm-workshop-highschool/blob/main/prompt_engineering_lab/prompt_engineering_lab.ipynb)

## Start here
- Students: `prompt_engineering_lab.ipynb` (open with the Colab badge above)
- Supporting files: `assets/`

Students call the model from inside the notebook. Each student needs a free API key from
[console.groq.com](https://console.groq.com), pasted into `API_KEY` in the second setup cell.
The first setup cell clones this repository on Colab so that `assets/` is available, and does
nothing when the notebook is run locally.

## Suggested setup before class
1. Confirm this repository is **public** — Colab cannot open a private repo without a GitHub login.
2. Open the Colab badge link yourself and run the whole notebook end to end with a real key.
3. Have students register for a Groq key in advance; it takes a few minutes each.
4. Tell students to click **File → Save a copy in Drive** before typing anything.
5. Display `assets/prompt_quality_checklist.md` during the activity.

Nothing in this lab is graded — the checklist is a diagnostic tool, not a scoring sheet.

## Core tasks
1. Public-domain song-style remix
2. Personal homepage
3. Personalized fitness/meal plan
4. Messy text → structured JSON
5. Standalone browser game

Challenges 1–4 are single-shot: one prompt, one response, no follow-up messages, so that
Prompt V1 and Prompt V2 can be compared without conversation history confounding the result.
Challenge 5 is deliberately multi-turn.
