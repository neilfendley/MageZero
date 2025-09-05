# MageZero: A Deck-Local AI Framework for Magic: The Gathering

### 1. High-Level Philosophy

MageZero is not a reinforcement learning (RL) agent in itself. It is a framework for training and managing deck-specific RL agents for Magic: The Gathering (MTG). Rather than attempting to generalize across the entire game with a monolithic model, MageZero decomposes MTG into smaller, more tractable subgames. Each deck is treated as a self-contained "bubble" that can be mastered independently using focused, lightweight RL techniques.

This approach reframes the challenge of MTG AI from universal mastery to local optimization. By training agents within constrained, well-defined deck environments, MageZero can develop competitive playstyles and meaningful policy/value representations without requiring LLM-scale resources.

---

### 2. Current Status: **Alpha (Actively in Development)**

The core infrastructure for MageZero is complete and undergoing testing. The full end-to-end pipeline—from simulation and data generation in Java to model training in PyTorch and back to inference via an ONNX model—is functional.

MageZero has successfully passed its **second conceptual benchmark**, demonstrating iterative improvement of the MCTS agent against a fixed heuristic opponent in a complex matchup (UW Tempo vs. Mono-Green). The current focus is now on **optimizing the simulation pipeline** and scaling further self-play experiments.

---

### 3. Core Components & Pipeline

MageZero's architecture is an end-to-end self-improvement cycle.

#### **Game Engine & Feature Encoding**

MageZero is implemented atop XMage, an open-source MTG simulator. Game state is captured via a custom `StateEncoder.java`, which converts each decision point into a high-dimensional binary feature vector.

* **Dynamic Feature Hashing**: This system supports a sparse, open-ended state representation while maintaining fixed-size inputs for the network. Features are dynamically assigned to slots in a preallocated bit vector (e.g., 200,000 bits) on first occurrence. A typical deck matchup utilizes a \~5,000 feature slice of this space.
* **Hierarchical & Abstracted Features**: The encoding captures not just card presence but also sub-features (like abilities on a card) and game metadata (life totals, turn phase). Numeric features are discretized, and cardinality is represented through thresholds. Sub-features pool up to parent features, creating additional layers of abstraction (e.g., a "green" sub-feature on a creature contributes to a "green permanents on the battlefield" count), providing a richer, more redundant signal for the model.

#### **Neural Network Architecture**

The model is a Multi-Layer Perceptron (MLP) designed to be lightweight but effective for the deck-local learning task.

* **Structure**: A massive, sparse embedding bag (for up to 200,000 features) feeds into a series of dense layers (512 -> 256) before splitting into two heads:

  * **Policy Head**: Predicts the optimal action (trained with Cross-Entropy Loss).
  * **Value Head**: Estimates the probability of winning (trained with Mean Squared Error). The target blends the MCTS root score (as in MuZero) with a discounted terminal reward.
* **Optimization**: The network uses a combination of Adam and SparseAdam optimizers. Training incorporates dropout layers for regularization.

#### **Initial Model Performance**

The network has proven capable of learning complex game patterns from relatively small datasets. The following results were achieved training the model to predict the behavior of AI agents in the UW Tempo vs. Mono-Green matchup.

| Training Data Source    | Sample Size | Engineered Abstraction | Policy Accuracy | Value Loss |
| ----------------------- | ----------- | ---------------------- | --------------- | ---------- |
| Minimax (UW Tempo only) | \~9,000     | Yes                    | 90+%            | <0.1       |
| Minimax (Both Players)  | \~9,000     | Yes                    | 88%             | <0.1       |
| MCTS (UW Tempo only)    | \~9,000     | Yes                    | 85%             | <0.15      |
| Minimax (UW Tempo only) | \~2,000     | Yes                    | 80%             | -          |
| Minimax (UW Tempo only) | \~2,000     | No                     | 68%             | -          |

---

### 4. Self-Play Results (as of Sept 2025)

Against a fixed minimax baseline (UW Tempo vs Mono-Green), MageZero improved from **16% → 30% win rate** over seven self-play generations. UW Tempo was deliberately chosen for testing because it is a difficult, timing-based deck — ensuring MageZero could demonstrate the ability to learn complex and demanding strategies.

**Win-rate trajectory**

| Generation         | Win rate |
| ------------------ | -------- |
| Baseline (minimax) | **16%**  |
| Gen 1              | 14%      |
| Gen 2              | 18%      |
| Gen 3              | 20%      |
| Gen 4              | 24%      |
| Gen 5              | 28%      |
| Gen 6              | 29%      |
| Gen 7              | **30%**  |

**Current Simulation Metrics**

* Games/hour (local, 13 CPU threads, 300-sim MCTS budget): \~150 games/hour
* Single-thread MCTS sims/sec: \~150
* 8-thread MCTS sims/sec: \~75 (limited by heavy heap usage)
* Target after XMage optimizations: \~1,000 games/hour

---

### 5. Critical Observations

Through experimentation, several key lessons have emerged:

* **Search Depth as a Catalyst**: Deeper MCTS search is crucial to allow the network to receive meaningful updates without being overwhelmed by noise. Shallow searches tend to produce unstable or misleading gradients.
* **Learning Speed and Depth**: An inverse relationship has been observed between the number of generations required per % improvement and the depth of search. Roughly, **doubling search depth makes the model learn almost twice as fast**.
* **Exploration Strategy**: Instead of Dirichlet noise, MageZero uses very soft temperature sampling (with a tunable temperature parameter) and occasionally resets priors. This balances stability and exploration while avoiding overconfidence in early policies.
* **Training Choices**:

  * Policy trained on decision states; value trained on all states.
  * Tighter PyTorch-based ignore list reduces active feature space to \~2,700.
  * Dropout layers improve regularization and generalization.

---

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
