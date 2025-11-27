# MageZero: A Deck-Local AI Framework for Magic: The Gathering

### 1. High-Level Philosophy

MageZero is not a reinforcement learning (RL) agent in itself. It is a framework for training and managing deck-specific RL agents for Magic: The Gathering (MTG). Rather than attempting to generalize across the entire game with a monolithic model, MageZero decomposes MTG into smaller, more tractable subgames. Each deck is treated as a self-contained environment that can be mastered independently using focused, lightweight RL techniques.

This approach reframes the challenge of MTG AI from universal mastery to local optimization. By training agents within constrained, well-defined deck environments, MageZero can develop competitive playstyles and meaningful policy/value representations without requiring LLM-scale resources. 

---

### 2. Current Status (October 2025): **Learning-MCTS agent implemented in Parralel AIvsAI environment in XMage**

The core infrastructure for MageZero is complete and undergoing testing. The full end-to-end pipeline from simulation and data generation in Java to model training in PyTorch and back to inference via local python server is functional.

If you are interested in contributing or running locally see the [setup guide]([url](https://github.com/WillWroble/MageZero/blob/main/setup_guide.md)). I am also always available at <willwroble@gmail.com>



---

### 3. Core Components & Pipeline

MageZero's architecture is an end-to-end self-improvement cycle.

#### **Game Engine & Feature Encoding**

MageZero is implemented atop XMage, an open-source MTG simulator. Game state is captured via a custom `StateEncoder.java`, which converts each decision point into a high-dimensional binary feature vector.

* **Dynamic Feature Hashing**: This system supports a sparse, open-ended state representation to handle all of the discrete artifacts and tokens MTG games can produce. This is done by use of a massive sparse Embedding Bag (2M features) with Weinberger style feature hashing. A typical 60card deck matchup utilizes a \~5,000 feature slice of this space. With usually around ~200 active features per state (after filtering redundent features) making chance of collision <0.01%. Since all decks share the same massive input space, overlapping deck feature slices allow for potential cross-deck learning.
* **Hierarchical & Abstracted Features**: The hashing captures not just card presence but also sub-features (like abilities on a card) and game metadata (life totals, turn phase). Numeric features are discretized, and cardinality is represented through thresholds. Sub-features pool up to parent features, creating additional layers of abstraction (e.g., a "green" sub-feature on a creature contributes to a "green permanents on the battlefield" count), providing a richer, more redundant signal for the model.

#### **Neural Network Architecture**

The model is a specialized Multi-Layer Perceptron (MLP) with a 2M dimension Embedding Bag input layer and outputs to 4 policy heads, and 1 value head.  

* **Structure**: 
  * **Embedding Bag**: 2M dimensions; uses 'SUM' (so gradients scale with active feature count) with sparseAdam, and usually has ~200 active binary feature indices.
  * **Embedding Layer** 512 dimensions; uses biases + batch norm with dropout layer (could theoretically be a shared embedding space for all states across all decks in MTG since input space is global)
  * **Hidden Layer** 256 dimensions; relu activation + biaes. (deck local embedding for policy + value)
  * **Policy Heads**: all deck local or matchup local
    * **Player Priority**: 128D; deck local; each logit corresponds to a priority action the Agent (PlayerA) can take (eg. activated abilities, casting spells). usually around ~20 logits are used per deck
    * **Opponent Priority**: 128D; opponent deck local; each logit corresponds to a priority action the opponent (PlayerB could take). when running MCTS vs MCTS games. both Agents share one network. and use each head.
    * **Targets**: 128D; matchup local; shared target space across both decks for all micro decisions involving targets. (this is used for selecting which attacking creature to use a blocker on). usually ~60 logits used per matchup
    * **Binary decisions** 2D: matchup local; shared binary space for all binary decisions made by either player. (this is used to select blockers and attackers sequentially)
  * **Value Head**: Estimates the probability of winning (trained with Mean Squared Error). The target blends the MCTS root score (as in MuZero) with a discounted terminal reward.
* **Optimization**: The network uses a combination of Adam and SparseAdam optimizers. Training incorporates dropout layers (p=0.3) for regularization.
* **Training**: all training samples are flagged with their decision type (player priority, opponent priority, target decision, binary decision). all sample types are trained together in mixed batches but policy gradients are gated to each sample's corresponding policy head. 

### 4. Self-Play Results (as of October 2025)

See Results file

**Current Simulation Metrics**

* Games/hour (local, 13 CPU threads, 300-sim MCTS budget): \~250 games/hour
* Single-thread MCTS sims/sec: \~150 (4ghz)
* 8-thread MCTS sims/sec: \~75 (limited by heavy heap usage)
* network single batched inferences/sec: (~100)
---

### 5. MCTS

MageZero uses the same combination of MCTS with Deep learning that Deep Mind's AlphaZero, Muzero and other variants have used to play Go, Chess, Shogi, and other strategy games at a superhuman level. Specifically we use the PUCT formula as was originally outlined in the AlphaZero paper. 

$$a^* = \arg\max_a \left[ Q(s,a) + c_{\text{puct}}  P(s,a)  \frac{\sqrt{N(s)}}{1 + N(s,a)} \right]$$


we use c = 1.0. but otherwise keep the formula the same. however there were many other modifications that needed to be made.

For one, unlike Chess, MTG has many different type of decision points. (priority, choosing targets, ordering triggers, attacking etc.) This is why we use a special policy head for each one, since all of these decisions can be game swinging and are highly learnable.

We also don't use and Dirchlet noise or temp sampling like the original AlphaZero authors did. Instead we found we were able to get stable network progression by increasing search depth since MTG already has a lot of inherent randomness. 

Another key difference was using a discount factor on the terminal label + blending it with the root Q value for more stable value targets for the network as employed by Muzero authors.

$$
v_{\text{target}}
= (1-\lambda)S_{\text{root}}
+
\lambda\gamma^{T}z_T
$$





we also use a special blend for learning opponent playstyle when playing against other types of AI (eg minimax). We use the MCTS visits the agent had for the opponent at that node + K virtual visits for the observed action:

$$
\tilde N(s,a) = N_{\text{pred}}(s,a) + k [a = a_{\text{obs}}]
\quad,\qquad
\pi_{\text{opp}}(a \mid s) = \frac{\tilde N(s,a)}{\sum_b \tilde N(s,b)}
$$



### 6. Challenges

MageZero faces several research challenges that shape future development:

* **Imperfect Information**: Unlike games like Go or Chess, Magic: The Gathering is a game of imperfect information where the opponent's hand and library are hidden. Handling this requires new methods, potentially drawing on MuZero-style learned dynamics models.

* **Long-Horizon & Weak Reward Signals**: The consequences of an early decision may not become apparent for many turns. Credit assignment remains a core challenge and is why I feel the need for a high quality bootstrap.

* **Simulation Throughput**: MCTS simulations are computationally expensive and XMage is heap intensive. Optimizing throughput remains a persistent challenge.

* **Evaluation Methodology**: No gold standard exists for MTG AI benchmarking. Win rate against fixed opponents remains the main reference metric.

---

### 7. Future Goals

1. **LLM-Based Bootstrap Agent**: Replace the minimax bootstrap with a stronger LLM-based agent to provide higher-quality priors and value signals.
2. **AI vs AI Simulation Framework**: Build a general framework within XMage for fast AI vs AI simulations, enabling MageZero and other MTG AI projects to scale evaluation and training.
3. **Clean Up & Refactor**: Solidify the existing codebase for stability and readability.
4. **Micro-Decision Policies**: Extend the learning process to cover fine-grained decisions such as targeting.
5. **Simulation Efficiency**: Develop less memory intensive Java simulations that approach \~1,000 games/hour.
6. Consolidate/containerize the entire pipeline with OpenAI gym or similiar. This is for use of HPC clusters and ease of distribution/collaboration.

---

### 8. Sources and Inspirations

MageZero draws from a range of research traditions in reinforcement learning and game theory.

* **AlphaZero & MCTS**: The core self-play loop, use of a joint policy/value network, and the PUCT algorithm for tree search are heavily inspired by the work on AlphaGo and AlphaZero.

  * Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). *Mastering the game of Go without human knowledge*. Nature, 550(7676), 354–359.
  * Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Science, 362(6419), 1140–1144.
* **MuZero**: Inspiration for blending MCTS root scores with discounted rewards and exploring the potential of learned dynamics models for handling hidden information and scaling simulations.

  * Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. Nature, 588, 604–609.
* **Feature Hashing**: The dynamic state vectorization method is an application of the hashing trick, a standard technique for handling large-scale, sparse feature spaces in machine learning.

  * Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009). *Feature Hashing for Large Scale Multitask Learning*. Proceedings of the 26th Annual International Conference on Machine Learning.
* **Curriculum Learning**: Though currently on the backburner, the initial concept for a "minideck curriculum" is based on the principle of gradually increasing task complexity to guide the learning process.

  * Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum learning*. Proceedings of the 26th Annual International Conference on Machine Learning.
