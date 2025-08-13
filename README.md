***

# MageZero: A Deck-Local AI Framework for Magic: The Gathering

### 1. High-Level Philosophy

MageZero is not a reinforcement learning (RL) agent in itself. It is a framework for training and managing deck-specific RL agents for Magic: The Gathering (MTG). Rather than attempting to generalize across the entire game with a monolithic model, MageZero decomposes MTG into smaller, more tractable subgames. Each deck is treated as a self-contained "bubble" that can be mastered independently using focused, lightweight RL techniques.

This approach reframes the challenge of MTG AI from universal mastery to local optimization. By training agents within constrained, well-defined deck environments, MageZero can develop competitive playstyles and meaningful policy/value representations without requiring LLM-scale resources.

---

### 2. Current Status: **Alpha (Actively in Development)**

The core infrastructure for MageZero is complete and undergoing testing. The full end-to-end pipeline—from simulation and data generation in Java to model training in PyTorch and back to inference via an ONNX model—is functional.

The current focus is on **optimizing the simulation pipeline** and executing the **second conceptual benchmark**: demonstrating that the MCTS agent can learn and iteratively improve its performance against a fixed, heuristic opponent in a complex matchup (UW Tempo vs. Mono-Green).

---

### 3. Core Components & Pipeline

MageZero's architecture is an end-to-end self-improvement cycle.

#### **Game Engine & Feature Encoding**
MageZero is implemented atop XMage, an open-source MTG simulator. Game state is captured via a custom `StateEncoder.java`, which converts each decision point into a high-dimensional binary feature vector.

* **Dynamic Feature Hashing**: This system supports a sparse, open-ended state representation while maintaining fixed-size inputs for the network. Features are dynamically assigned to slots in a preallocated bit vector (e.g., 200,000 bits) on first occurrence. A typical deck matchup utilizes a ~5,000 feature slice of this space.
* **Hierarchical & Abstracted Features**: The encoding captures not just card presence but also sub-features (like abilities on a card) and game metadata (life totals, turn phase). Numeric features are discretized, and cardinality is represented through thresholds. Sub-features pool up to parent features, creating additional layers of abstraction (e.g., a "green" sub-feature on a creature contributes to a "green permanents on the battlefield" count), providing a richer, more redundant signal for the model.

#### **Neural Network Architecture**
The model is a Multi-Layer Perceptron (MLP) designed to be lightweight but effective for the deck-local learning task.

* **Structure**: A massive, sparse embedding bag (for up to 200,000 features) feeds into a series of dense layers (512 -> 256) before splitting into two heads:
    * **Policy Head**: Predicts the optimal action (trained with Cross-Entropy Loss).
    * **Value Head**: Estimates the probability of winning (trained with Mean Squared Error).
* **Optimization**: The network uses a combination of Adam and SparseAdam optimizers.

#### **Initial Model Performance**
The network has proven capable of learning complex game patterns from relatively small datasets. The following results were achieved training the model to predict the behavior of AI agents in the UW Tempo vs. Mono-Green matchup.

| Training Data Source | Sample Size | Engineered Abstraction | Policy Accuracy | Value Loss |
| :--- | :--- | :--- | :--- | :--- |
| Minimax (UW Tempo only) | ~9,000 | Yes | 90+% | <0.1 |
| Minimax (Both Players) | ~9,000 | Yes | 88% | <0.1 |
| MCTS (UW Tempo only) | ~9,000 | Yes | 85% | <0.15 |
| Minimax (UW Tempo only) | ~2,000 | Yes | 80% | - |
| Minimax (UW Tempo only) | ~2,000 | No | 68% | - |

#### **MCTS Self-Improvement Loop**
The training process is a cycle of play, learning, and improvement, based on the AlphaZero methodology.

1.  **Bootstrapping (Gen 0)**: We simulate ~250 games where the agent plays against a baseline heuristic minimax agent to generate an initial dataset (~9,000 states). The neural network is trained on this data to create the "Gen 0" model.
2.  **MCTS-Guided Play**: The Gen 0 model is integrated into a custom MCTS agent. The agent again plays against the same fixed minimax opponent. It uses the network's policy/value predictions and a PUCT formula with Dirichlet noise to guide its search and ensure robust exploration.
3.  **Data Collection**: Game states and the final MCTS visit counts (which become the new, improved policy labels) are saved to a replay buffer. The final game outcome provides the value label.
4.  **Retraining**: The model is retrained on data from the replay buffer, creating the "Gen 1" model.
5.  **Iteration**: The process repeats. Each new generation of the model is trained on data produced by its predecessor, allowing it to iteratively refine its strategy against the static opponent.

---

### 4. Current Progress & Benchmarks

MageZero's development is structured around a series of conceptual benchmarks.

#### **Benchmark 1: Model Viability**
* **Goal**: Prove the neural network can learn to predict AI agent behavior in a specific matchup with high accuracy.
* **Result**: **Success**. The model achieved >90% policy accuracy, validating the feature encoding and network architecture.

#### **Benchmark 2: Iterative MCTS Improvement**
* **Goal**: Demonstrate that the self-play loop can produce a tangible increase in win rate over successive generations against a fixed opponent.
* **Status**: **In Progress**. The pipeline is functional but requires optimization. Initial test runs are underway:
    * **Gen 0 (trained on minimax)**: ~8% win rate
    * **Gen 1**: ~6% win rate
    * **Gen 2**: ~10% win rate
    * **Analysis**: The initial dip in performance is expected. The Gen 0 model is trained on a weak agent. The introduction of MCTS with Dirichlet noise forces exploration, which can degrade performance as the agent moves away from the deterministic policy it was trained on. Subsequent generations are expected to show a clear upward trend as the agent refines its strategy based on its own, more sophisticated search.

#### Benchmark 3: Multi-Deck Learning
* **Goal**: Demonstrate that **learning** can be shared or transferred between different decks, realizing the full vision of MageZero.
* **Status**: **Planned**.
* This will begin after Benchmark 2 is definitively passed.

---

### 5. Core Challenges & Roadmap

* **Imperfect Information**: Unlike games like Go or Chess, Magic: The Gathering is a game of imperfect information where the opponent's hand and library are hidden. Standard Monte Carlo Tree Search (MCTS) is designed for perfect information games. Applying it here requires handling uncertainty, as the agent must make decisions without knowing the opponent's full state. This complicates the search process, which must account for a wide range of possible hidden cards rather than a single, known game state.
* **Long-Horizon & Weak Reward Signals**: In MTG, the consequences of an early decision may not become apparent for many turns. This creates a difficult credit assignment problem, as the terminal reward (winning or losing) is often too delayed to effectively guide the agent. The immediate, turn-by-turn board state can be a poor indicator of success, making it challenging for the agent to learn winning long-term strategies from a weak signal.
* **Need for a Flawed Bootstrap**: Given the weak reward signal, starting with a purely random policy is intractable. It is necessary to bootstrap the learning process with an initial policy, even a flawed one like that from the heuristic minimax agent. This provides an essential, albeit imperfect, initial gradient to guide the first generation of the MCTS agent, giving it a starting point from which to begin exploration and iterative improvement.
* **Simulation Throughput**: MCTS simulations are computationally expensive. The primary bottleneck is optimizing the simulation pipeline to better leverage multi-core architectures and reduce memory overhead, which is critical for generating training data at a reasonable pace.
* **Evaluation Methodology**: No gold standard exists for MTG AI benchmarking. Win rate against a fixed opponent is the current primary metric for tracking iterative improvement.

My immediate goal is to complete Benchmark 2 before August 20th, 2025.

1.  **Clean Up & Refactor**: Solidify the existing codebase for stability and readability.
2.  **Optimize Pipeline**: Refactor the MCTS implementation to enable greater parallelization, significantly reducing the time required to generate training batches.
3.  **Test & Tune**: With an optimized pipeline, perform extensive testing of the MCTS improvement loop to achieve a consistent increase in win rate across generations.

If MageZero passes this second benchmark, I plan to commit to the project long-term and begin seeking collaborators to help actualize its potential as a functional application for the MTG community.

---
### 6. Sources and Inspirations

MageZero draws from a range of research traditions in reinforcement learning and game theory.

* **AlphaZero & MCTS**: The core self-play loop, use of a joint policy/value network, and the PUCT algorithm for tree search are heavily inspired by the work on AlphaGo and AlphaZero.
    * Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). *Mastering the game of Go without human knowledge*. Nature, 550(7676), 354–359.
    * Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Science, 362(6419), 1140–1144.
* **Feature Hashing**: The dynamic state vectorization method is an application of the hashing trick, a standard technique for handling large-scale, sparse feature spaces in machine learning.
    * Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. (2009). *Feature Hashing for Large Scale Multitask Learning*. Proceedings of the 26th Annual International Conference on Machine Learning.
* **Curriculum Learning**: Though currently on the backburner, the initial concept for a "minideck curriculum" is based on the principle of gradually increasing task complexity to guide the learning process.
    * Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum learning*. Proceedings of the 26th Annual International Conference on Machine Learning.
