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
## Run the simulator

The native runners use Linux, Python 3.11, MuJoCo and EGL. Use a CUDA-capable EGL installation for cameras. On Ubuntu, install Git, ffmpeg, the OpenGL/EGL runtime and the `fonts-noto-core` package. Run the commands below from a clone of this repository.

```bash
python3.11 -m venv ../leash-env
source ../leash-env/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r benchmark/requirements-sim.txt
python tools/prepare_runtime.py ../leash-runtime
cd ../leash-runtime
source tools/activate.sh
python -m pip install -e upstream/RPent-reproduce-libero
python -m benchmark.native P3-01 --seed 0
```

The preparation command creates a separate Git worktree, binds archived MuJoCo asset paths to its location, writes the LIBERO configuration and checks out the two upstream revisions recorded in the script. Keep that worktree at the same path. Pinned upstream clones can also be supplied with `--upstream-cache`. The dependency versions come from the evaluation environment. Validation used those installed dependencies.

**Registered starts.** The benchmark includes trials that begin with an object in the gripper. The two CSV files share the keys `task, planner, method, seed`. Of the 3,312 counted trials, 332 used the registered tabletop pre-grasp initializer. The remaining 2,980 used ordinary starts. The settings table also records the protocol version, policy and task-instruction revision. To reconstruct a particular starting state, supply the exact trial key and a fresh output directory.

```bash
python -m benchmark.initial_condition E1-01 --seed 0 --planner gpt-6-astra --method LEASH/Jev --output runs/preheld-demo
```

**Exercise the execution boundary.** The following integration fixture runs a scripted planner and reviewer through the native simulator. It needs no provider credentials. Use fresh output directories for each invocation.

```bash
python -m research.v4.bridge.run_b0 --leash-instance E1-01 --seed 0 --method B0 --backend scripted --phase smoke --output runs/e1-baseline
python -m research.v4.bridge.run_b0 --leash-instance E1-01 --seed 0 --method Ours --backend scripted --phase smoke --output runs/e1-leash
```

Inspect `episode.json` for `scripted_business_valid` and `scripted_tools.json` for the commands. The LEASH run also writes gate decisions. These are deterministic integration fixtures and are excluded from HR and SSR. The counted model outcomes in the first section come from the original experiments.

**Source map.**

| Component | Location |
| --- | --- |
| Native task models, evaluators and scene bindings | `research/v4/tasks`, `research/v4/candidates`, `research/v4/bridge` |
| Shared dispatcher, lease and event registration | `research/v4/leash_v43` |
| Confidence gates and public observation packet | `research/v4/development/confidence_gate_dev_20261006` |
| Transition checks and reviewer implementations | `research/v4/development/g3_transition_dev_20261019`, `decision_gate_dev_20261019`, `score_only_g3_030_20261021` under the same parent |
| Reviewed policy and pre-held initialization | `research/v4/development/user_review_revision_20261026` |
| Dobot hardware boundary and typed reviewer port | `hardware/dobot` |

Live experiments additionally require the selected planner and reviewer services, the Pi0.5 LIBERO checkpoint and SAM3 weights. The experiment runner expects `checkpoints/pi05-libero130/` and `checkpoints/sam3/sam3.pt`. Bring your own provider access. Cost, service revisions and stochastic model responses can change. The private campaign scheduler, account material and raw requests are excluded. The archived labels support exact metric reproduction independently of these services.

## Exercise the physical LEASH port

In the prepared environment, run the physical bridge tests with a mock driver.

```bash
python -m unittest discover -s hardware/dobot/tests -v
```

The 21 tests exercise the shared gate boundary, including withheld commands, interruption, stale observations and command acknowledgements. A hardware integration supplies its own driver and gripper calibration to `hardware.dobot.bridge.build_port`. For the production primitive source use `--physical-primitives` when preparing a new worktree. The typed reviewers read `OPENAI_API_KEY` or `TYPESAFE_API_KEY` from the process environment. For the physical trials themselves, use the inventory in the paper and the paired camera images on the project website.
