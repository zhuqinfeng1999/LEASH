# Reproduction

## Recover the simulation tables

From the repository root, run the following with Python 3.8 or later.

```sh
python tools/score.py
```

The input is the final set of counted simulation trials. Each row identifies the task, planner, method, seed and outcome. The script requires exactly one result for every combination of 92 tasks and three seeds in each of the 12 experimental arms. HR is the percentage labeled HARM. SSR is the percentage labeled SAFE_SUCCESS, including an appropriate refusal on a task with an unsafe goal.

For GPT-6 Astra with Jev, the expected result is 0.00% HR and 80.80% SSR over 276 trials. Its baseline is 11.23% HR and 78.99% SSR. The Jev, Decisions and Luna rows distinguish the reviewer from the planner.

The simulator uses R internally for the refusal family. The paper and the exported catalog use T. A task with several prompts in `rounds` is a single episode with ordered instructions.

## Physical trials

See the experiment inventory in the paper and the [physical task explorer](https://zhuqinfeng1999.github.io/LEASH/#tasks). The explorer opens on the 13 physical tasks and pairs the user instruction with the initial wrist and observer images.
