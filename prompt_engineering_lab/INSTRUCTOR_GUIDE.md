# 45-Minute Prompt Engineering Lab — Instructor Guide

## Learning objectives
Students will:
1. Treat prompting as an iterative design process rather than a one-shot trick.
2. Use context, delimiters, constraints, examples, and explicit output formats.
3. Apply in-context learning through reference examples.
4. Use planning, critique, and verification steps for complex tasks.
5. Compare prompt versions with observable criteria.

This lab is ungraded. There are no points or scores; the checklist in `assets/prompt_quality_checklist.md`
is for diagnosing prompts, and students should be told up front that a failed first output is expected material to work with.

## Recommended timing
- 0–5 min: Frame the challenge; form pairs; choose one task.
- 5–10 min: Run a deliberately weak baseline prompt and diagnose failures.
- 10–25 min: Build Prompt V1 using the prompt canvas.
- 25–35 min: Test, diagnose, and revise to Prompt V2.
- 35–42 min: Optional second task or final polish.
- 42–45 min: Two fast demos and reflection.

For a group that moves quickly, let teams choose two tasks. Trying three substantial tasks in 45 minutes
will usually reduce the quality of iteration.

## Five core choices
1. Public-domain song-style remix.
2. Single-file personal homepage.
3. Personalized fitness and meal-planning assistant.
4. Messy text to validated structured JSON.
5. Single-file browser game.

All five are ordinary tasks of equal standing — no task is an extra-credit or bonus track.

## Prompt canvas
- ROLE: Who should the model act as?
- GOAL: What concrete artifact should it produce?
- CONTEXT: What source material or user data is relevant?
- CONSTRAINTS: What must it do or avoid?
- EXAMPLES: What demonstrations establish the pattern?
- PROCESS: Ask it to plan, draft, check, and revise.
- OUTPUT CONTRACT: Exact sections, schema, code fence, length, etc.
- VERIFICATION: How should it test or audit the result?

## Important wording about reasoning
Rather than demanding hidden “chain of thought,” ask for:
- a brief plan,
- a concise rationale,
- a checklist-based self-review,
- explicit calculations or evidence when needed,
- and a final answer separated from the verification report.

## Facilitation questions
- What did the model have to guess?
- Which instruction had the biggest effect?
- Did your example teach content, format, or both?
- What failure persists even after your revision?
- Can another team reproduce a similar-quality output from your prompt?

## Safety notes
- Fitness content is educational, moderate, and age-appropriate; no diagnoses, supplements, or crash diets.
- Song task uses public-domain references and high-level traits; students should not request direct imitation of a living artist.
- Personal data are entirely fictional.
