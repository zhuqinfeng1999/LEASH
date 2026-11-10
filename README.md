<div align="center">
  <img src="media/leash.png" width="96" alt="The LEASH mascot">
  <h1>LEASH: Stay Safe in Agentic Robotics</h1>
  <p><b>A safety harness for robot planners and 105 tasks to test it.</b></p>
  <p><a href="https://zhuqinfeng1999.github.io/LEASH/">Project page</a> · <a href="#project-video">Watch the overview</a> · <a href="docs/reproduction.md">Reproduce the experiments</a></p>
</div>

## Abstract

A robot can follow an instruction exactly and still cause harm. When asked to put a phone in a bowl, for example, it needs to notice the water. **RoboRisk** tests this kind of judgment in closed loop across 92 simulated tasks and 13 physical tasks. We evaluate six foundation models and two robot-specific APIs across five kinds of risk, from unsafe requests to hazards that appear during motion.

**LEASH**, the Lease-Enforced Agentic Safety Harness, checks the task, the proposed action and the point just before an irreversible operation. The planner and all three gates use the same safety policy and ground decisions in current camera observations. In our experiments, LEASH reduced the harm rate for all six foundation models while largely preserving task completion. For hardware setups and physical trials, see the experiment inventory in the paper and the paired wrist and third-person images in the [task explorer](https://zhuqinfeng1999.github.io/LEASH/#tasks).

## Project video

[![Watch the 3-minute 19-second project overview](media/overview.jpg)](media/overview.mp4)

[Open or download the English overview (3:19)](media/overview.mp4)
