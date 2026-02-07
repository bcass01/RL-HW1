# COMPSCI 590.07 Assignment 1: Written Questions
**Name:** Bernard Cassidy

## Section 2

### 2.1 Why is the value function not 0 in all places but the rewarding states if our policy almost always goes UP?
The value function is greater than zero because there is a chance of “slipping”, or moving right or left instead of up, when the “up” action is chosen. This means that there is a chance of reaching the top-right rewarding state by only choosing the up action.

### 2.2 Why is the value of state index 1 smaller than state index 4?
When we are in state 1, there is a chance of reaching state 6 by slipping. For example, you could slip to the right from state 1 to state 2, move up to state 5, and then slip to the right again to state 6. However, there is no chance of reaching state 6 from state 4 because it is impossible to move down by slipping during the “up” action. Since state 4 has a barrier on both sides, this means that we will eventually move up to the top row, and then in the top row we can only move side to side.

### 2.3 Does this value function tell us anything about what might be the optimal policy?
Yes. States that have a higher chance of reaching the goal state, even if only by accident, have a higher value than states that give us a chance of reaching the penalty state. This means that a policy of taking a path from states with the lowest value to those with the highest value will likely give us a better chance of reaching the goal state and avoiding the penalty state, even if that value is calculated using a suboptimal policy.

---

## Section 3

### 3.1 How many iterations did it take for policy iteration to converge?
It took three iterations for the policy to converge.

---

## Section 4

### 4.1 At what iteration in `value_iteration2` does the action at state 7 change from its initialized action?
The value of `pi[7]` changes at i=3.

### 4.2 How would you interpret this? Why is it that the policy for state 7 changes at this value of i?
This happens because our `lookahead` function looks one step ahead, and it takes three steps to reach state 10 from state 7. On the first iteration, the value of stage 9 increases because of its proximity to stage 10. On the second, the value of stage 8 increases because the value of stage 9 has been “recognized” by our algorithm. On the third iteration, we “see” that stage 7 is next to the “newly”-valuable stage 8, and the optimal policy changes to one that directs us towards stage 8 from stage 7,

---

## Section 5

### 5.1 Explain the difference in policy between this policy and the policy learnt for $\gamma=0.9$ in the original MDP.
This policy chooses actions that provide a less direct route to the goal state, but eliminate the possibility of reaching the penalty state. Since our new gamma value is much higher, we have a much lower discount for future states. This means that routes that take a longer time to reach the goal state, but do not risk reaching the penalty state are more valuable in this MDP than in the original MDP.

### 5.2 Explain the difference in policy... why is it that the optimal policy for the agent at some states is different across the two policies?
This policy takes actions that will reach the goal state faster but run a higher risk of accidentally reaching the penalty state. Since our gamma value is lower, we take on a much larger penalty for each step we take reaching the goal state. This means that time (or number of steps) becomes a much more significant factor when calculating the value of each state and action. Therefore the optimal action at stage 1 becomes moving to the right, which gives us a greater chance of reaching the goal state faster even though it also moves us towards the penalty state.

---

## Section 7

### 7.1 Try different values of `mdp.gamma`. What is the relationship between `mdp.gamma` and the convergence iterations number required to stop?
There is a positive relationship: as $\gamma$ increases (approaches 1), the number of iterations required to converge increases. For a low gamma, the  is short; values stabilize quickly.
For a low $\gamma$ (e.g., 0.1), the program is "myopic." It cares mostly about immediate rewards and rewards that take a long time to reach are essentially worthless, so the values stabilize very quickly. For a high $\gamma$ (e.g., 0.99), the program is "far-sighted." Rewards obtained far in the future still have a significant impact on the current state's value. Information must propagate back and forth across the grid many times before the values settle down.

### 7.2 If the algorithm terminates because $\delta < \theta$, does it mean that we have found the exact $U^*$? Why is the contraction property essential...?
We have not found the exact $U^*$, but we have found an approximation $U_k$ that is within a bounded error of the true solution.

The Contraction Property $\| U_{k+1} - U^* \| \leq \gamma \, \| U_k - U^* \|$ is essential because:
1.  It guarantees convergence, ensuring that the sequence of values actually moves toward a unique fixed point $U^*$.
2.  It bounds the error, allowing us to mathematically prove that when the change $\delta$ is small, the distance to the true solution is also small. Without this property, a small $\delta$ would not guarantee that we are close to the optimal solution.

---

## Section 8

### Proof

Let the optimal Q-function with respect to the true rewards $R$ be:
$$Q^\star(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) U^\star(s')$$

Let the approximate Q-function, computed using the estimated rewards $\hat{R}$ and approximate values $U_k$, be:
$$\hat{Q}_k(s,a) = \hat{R}(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) U_k(s')$$

The policy $\pi$ is greedy with respect to $U_k$ (using $\hat{R}$), meaning $\pi(s) = \operatorname{argmax}_a \hat{Q}_k(s,a)$.
The optimal policy $\pi^\star$ is greedy with respect to $U^\star$, meaning $\pi^\star(s) = \operatorname{argmax}_a Q^\star(s,a)$.

We first bound the difference between the true optimal Q-values and our approximate Q-values.
$$|Q^\star(s,a) - \hat{Q}_k(s,a)| = \left| R(s,a) - \hat{R}(s,a) + \gamma \sum_{s'} P(s'|s,a)(U^\star(s') - U_k(s')) \right|$$

Using the Triangle Inequality and the bounds given in the problem ($|\hat{R}-R| \le \delta - \varepsilon$ and $|U_k - U^\star| \le \varepsilon$):
$$|Q^\star(s,a) - \hat{Q}_k(s,a)| \le |R(s,a) - \hat{R}(s,a)| + \gamma \sum_{s'} P(s'|s,a) |U^\star(s') - U_k(s')|$$
$$|Q^\star(s,a) - \hat{Q}_k(s,a)| \le (\delta - \varepsilon) + \gamma \varepsilon$$

Let $\Delta_{step}$ denote this single-step error bound:
$$\Delta_{step} = \delta - \varepsilon(1 - \gamma)$$

We calculate the loss incurred by choosing action $\pi(s)$ instead of $\pi^\star(s)$ at state $s$:
$$\text{Loss}(s) = Q^\star(s, \pi^\star(s)) - Q^\star(s, \pi(s))$$

We add and subtract $\hat{Q}_k(s, \pi^\star(s))$ and $\hat{Q}_k(s, \pi(s))$:
$$= Q^\star(s, \pi^\star(s)) - \hat{Q}_k(s, \pi^\star(s)) + \hat{Q}_k(s, \pi^\star(s)) - \hat{Q}_k(s, \pi(s)) + \hat{Q}_k(s, \pi(s)) - Q^\star(s, \pi(s))$$

1.  $Q^\star(s, \pi^\star(s)) - \hat{Q}_k(s, \pi^\star(s)) \le \Delta_{step}$
2.  $\hat{Q}_k(s, \pi^\star(s)) - \hat{Q}_k(s, \pi(s)) \le 0$ (Since $\pi$ maximizes $\hat{Q}_k$)
3.  $\hat{Q}_k(s, \pi(s)) - Q^\star(s, \pi(s)) \le \Delta_{step}$

Thus, the single-step loss is bounded by:
$$U^\star(s) - Q^\star(s, \pi(s)) \le 2\Delta_{step}$$


The value difference satisfies the recurrence relation:
$$U^\star(s) - U_\pi(s) = U^\star(s) - Q^\star(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s)) (U^\star(s') - U_\pi(s'))$$

Taking the max norm $\|\cdot\|_\infty$:
$$\|U^\star - U_\pi\|_\infty \le 2\Delta_{step} + \gamma \|U^\star - U_\pi\|_\infty$$
$$(1 - \gamma) \|U^\star - U_\pi\|_\infty \le 2\Delta_{step}$$
$$\|U^\star - U_\pi\|_\infty \le \frac{2\Delta_{step}}{1 - \gamma}$$

Substitute $\Delta_{step} = \delta - \varepsilon(1 - \gamma)$ back into the equation:
$$\|U^\star - U_\pi\|_\infty \le \frac{2(\delta - \varepsilon(1 - \gamma))}{1 - \gamma}$$

Separating the terms:
$$\|U^\star - U_\pi\|_\infty \le \frac{2\delta}{1 - \gamma} - 2\varepsilon$$
