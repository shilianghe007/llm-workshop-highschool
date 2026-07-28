# Prompt Quality Checklist

Nothing in this lab is graded. Use this list to diagnose a prompt, not to score it.

1. **Goal clarity** — Is the requested outcome unambiguous?
2. **Relevant context** — Does the prompt supply the information needed, without unrelated noise?
3. **Constraints and safety** — Are requirements, boundaries, and edge cases explicit?
4. **Examples / in-context learning** — Are useful demonstrations or reference materials supplied?
5. **Output contract and verification** — Is the format specified, and does the model check its work?

## Iteration protocol
- Run the baseline prompt.
- Identify the largest failure.
- Change only one or two prompt components.
- Run again and compare.
- Keep evidence: prompt version, output, which checklist items were weak, and a one-sentence diagnosis.
