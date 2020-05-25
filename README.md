# STRATA

STRATA is a task assignment algorithm for heterogeneous multi-agent systems [1].

*Abstract*: Large teams of heterogeneous agents have the potential to solve complex multi-task problems that are intractable for a single agent working independently. However, solving complex multi-task problems requires leveraging the relative strengths of the different kinds of agents in the team. We present Stochastic TRAit-based Task Assignment (STRATA), a unified framework that models large teams of heterogeneous agents and performs effective task assignments. Specifically, given information on which traits (capabilities) are required for various tasks, STRATA computes the assignments of agents to tasks such that the trait requirements are achieved. Inspired by prior work in robot swarms and biodiversity, we categorize agents into different species (groups) based on their traits. We model each trait as a continuous variable and differentiate between traits that can and cannot be aggregated from different agents. STRATA is capable of reasoning about both species-level and agent-level variability in traits.

## Usage

To get started:

```
python run_STRATA.py
```

## Acknowledgement
The package provided here was built on top of the [source code](https://github.com/amandaprorok/diversity) for prior work by Amanda Prorok [2].

## References

[1] H. Ravichandar, K. Shaw, S. Chernova. STRATA: Unified Framework for Task Assignments in Large Teams of Heterogeneous Agents, *Journal of Autonomous Agents and Multi-Agent Systems (J-AAMAS)*, vol. 34, no. 38, 2020. [[preprint](https://arxiv.org/abs/1903.05149)] [[publisher](https://link.springer.com/epdf/10.1007/s10458-020-09461-y?sharing_token=mq7qZiSsareAzPMyZf6Hq_e4RwlQNchNByi7wbcMAY4DjhFgCwmY9Z16K1oYwkIdCWl0p-1GLKzZAYNXqzVpGPJQxcfBZqMXmui70HTgTtBe_vCObYhHJrwiti4HONkHThwW48PhZ8GAWdJH9ozTErKg-o5-_UvtK_C4HVUAqDI%3D)]

[2] A. Prorok, M. A. Hsieh, and V. Kumar. The Impact of Diversity on Optimal Control Policies for Heterogeneous Robot Swarms. IEEE Transactions on Robotics (T-RO), vol. 33, no. 2, 2017 [[preprint](https://scalar.seas.upenn.edu/wp-content/uploads/2018/01/Prorok_TRO_2016.pdf)]
