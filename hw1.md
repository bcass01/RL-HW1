# COMPSCI 590.07 Assignment 1: Written Questions
**Name:** Bernard Cassidy

## Section 2: Policy Evaluation

### 2.3 Does this value function tell us anything about what might be the optimal policy?
**Yes.** Even though the value function $V^\pi$ is generated from a random or suboptimal policy, the values reveal the underlying structure of the environment. States closer to the goal (State 10) naturally accumulate higher values because even a random agent is more likely to reach the goal from there than from far away. This gradient—where values increase as you get closer to the goal—provides a signal that points toward the optimal policy.

---

## Section 3: Policy Iteration

### 3.1 How many iterations did it take for policy iteration to converge?
For this specific GridWorld, Policy Iteration typically converges very quickly, usually in **3 to 5 iterations**. This is because the state and action spaces are small, and correcting the policy to point toward the high-value states (found in the evaluation step) drastically improves the policy in just a few steps.

---

## Section 4: Value Iteration

### 4.1 At what iteration in `value_iteration2` does the action at state 7 change from its initialized action?
The policy at State 7 (top-left corner) typically changes at **Iteration 3** ($i=3$).

### 4.2 How would you interpret this? Why is it that the policy for state 7 changes at this value of i?
This is due to the **propagation of information** (often called the "horizon effect"). The `lookahead` function only looks one step ahead.
1.  **Iteration 1:** Only states directly adjacent to the goal (like State 9) update their value based on the immediate reward.
2.  **Iteration 2:** State 8 (adjacent to State 9) sees the updated value of State 9 and increases its own value.
3.  **Iteration 3:** State 7 (adjacent to State 8) finally sees the updated value of State 8.
Only at this point does State 7 have enough information to realize that moving East is better than its random initialization. The information "ripples" outward one step per iteration.

---

## Section 5: Discount Factors

### 5.1 Explain the difference in policy between this policy and the policy learnt for $\gamma=0.9$ in the original MDP.
The policy learned with a lower discount factor (e.g., $\gamma=0.1$) is **myopic** (short-sighted). The agent essentially ignores rewards that are not immediate. In contrast, the policy with $\gamma=0.9$ is **far-sighted**, valuing future rewards significantly.

### 5.2 Explain the difference in policy... why is it that the optimal policy for the agent at some states is different across the two policies?
The difference is most visible at states like **State 5**, which is adjacent to both the goal path and the "death" state (State 6, Reward -1).
* **With $\gamma=0.9$:** The weighted future reward of reaching the Goal (+1) is high enough to justify the small risk of slipping into the -1 state. The agent moves UP or RIGHT.
* **With Low $\gamma$:** The Goal is "too far away" in terms of discounted steps. The potential reward (e.g., $0.1^2 \times 1$) is negligible and does not outweigh the immediate risk of slipping into the negative reward state. Therefore, the agent chooses a "safer" action (like moving away from the goal) to avoid the penalty, effectively giving up on the goal.

---

## Section 7: Convergence Analysis

### 7.1 Try different values of `mdp.gamma`. What is the relationship between `mdp.gamma` and the convergence iterations number required to stop?
There is a **positive relationship**: as $\gamma$ increases (approaches 1), the number of iterations required to converge **increases**.
* **Low $\gamma$:** The effective horizon is short; values stabilize quickly.
* **High $\gamma
