
## **1. System Requirements**

### **Hardware (minimum / recommended)**

* **Minimum:** any quad-core CPU + 16GB RAM
* **Recommended:** 8–16 cores + 32–64GB RAM (MCTS is CPU heavy)
* **GPU:** NVIDIA card w/ 8GB+ VRAM (for model training)
* **Disk:** ~10GB for XMage + logs + model checkpoints (40GB+ recommended)

### **Software**

* Windows or Linux
* Java 21+
* IntelliJ (strongly recommended for XMage)
* Python 3.10 or later
* PyTorch (with CUDA)
* Git

---

## **2. Repository Structure**

* `/` — PyTorch model code, dataset utilities, training scripts
* `/data/` — XMage generated HDF5 datasets per deck
* `/models/` — trained models per deck/ver
* `/stats_out/` — generated histograms and graphs from dataset files

---

## **3. Installing XMage fork**

### **3.1 Clone the XMage fork**

```
git clone https://github.com/WillWroble/mage
```

### **3.2 Import XMage into IntelliJ**

* Open IntelliJ → "Import Project" → select `/mage/`
* Make sure Maven syncs
* Ensure Java 21 is installed and selected as SDK

### **3.3 Build maven project**

see https://github.com/magefree/mage/wiki/Setting-up-your-Development-Environment

* increase compiler's heap size to 1500 or 3000 MB by Settings -> Build -> Compiler -> Build process heap size
* run maven clean and install 
* with Intellij maven: execute Mage root/lifecycle/clean and Mage root/lifecycle/install under maven project (M on right column)
* build the project (can take 30min+)
---

## **4. Running AI vs AI Simulations (Dataset Generation)**

### **4.1 How to add Decks**
* export the deck(s) you want to use as text from Moxfield, MTGO, MTGA or similar
* convert your deck as a `.dck` file either using XMage's client deckbuilder, or from a text file using `TxtToDckConversionTest.java`
* add the created .dck files to `\Mage.Tests\deck`
* sideboard not supported yet

### **4.2 Where to Configure Matchups**

Right now all RL self play sessions are executed from JUnit tests in `Mage.Tests/src/java/../AI/RL`
The root test harness is at `ParallelDataGenerator.java` most parameters you'll need to change will be static fields at the top of this file 
(replace the defaults as you see fit, especially `MAX_CONCURRENT_GAMES` - don't go beyond 8 to avoid cache contention unless you have a supercomputer)
Most relevant RL parameters are in this file, but if you want to change the more in-depth MCTS parameters, go to `ComputerPlayerMCTS.java` (the most relevant of
which being `MAX_TREE_VISITS` which is the budget for the tree search, only lower if games are way too slow) 

There are 2 main data generation configurations for RL right now:
* `SimulateRLvsMinimax.java` (recommended to start) this is for playing games against XMages minimax AI (mad bot). 
the mad bot is much faster and weaker and recommended to start with for initial training.
* `SimulateRLVsRL.java` (advanced) this is for playing games against another MCTS RL bot. this requires having 2 different MageZero network servers running
and is extremely Memory, CPU and GPU intensive. Right now we are trying to make a database of high quality pre-trained deck-local models for an initial opponent pool. 
for now, you will have to use your own models or models I've shared in the public drive folder.



### **4.3 Running Simulations**


* Make default run configuration for JUnit tests. this is hardware dependent, but it is recommended to set heap as high as possible
and use ZGC. 
to do this in Intellij: run configurations > edit configurations > edit config templates > 
JUnit > add VM options > `-Xms2g -Xmx24g -XX:+UseZGC --add-opens=java.base/java.lang=ALL-UNNAMED`

To start learning the pipeline I recommend fixing a single minimax matchup in `SimulateRLvsMinimax.java` to do this:

* in setup() at the top of the file, change `DECK_A` to be the name of the deck you want your RL agent to play. change `DECK_B` to be 
the opponent's deck. (decks names shouldn't include the .dck , example: 
```
    DECK_A = "my_imported_deck";
    DECK_B = "MTGA_MonoR;"
```
**PlayerA always refers to the player you are training the network for, PlayerB is always the opponent**
* also optionally change RL parameters and settings here as you see fit. (I recommend keeping defaults and keeping `DONT_USE_POLICY` to true until you network's 
priority A accuracy is >70% which can take 1-3 1000 game gens). Always have `DONT_USE_POLICY_TARGET` set to true for now 
* go to `createTrainDataSet()` change `NUM_GAMES_TO_SIMULATE` as you see fit and run the JUnit test
* since no inference server is running the agent will fall back to offline mode, which uses a heuristic value function and uniform policy
* *Optional-Recommended* - get `grep` extension for IntelliJ and use it on output console to filter output per thread
* When this completes it will generate `.hdf5` files under `Mage.Tests/training` make sure to save these files before starting a new run (they are overridden each time)


## **5. Python Environment Setup (Training Pipeline)**

### **5.1 Clone MageZero repo**

```
git clone https://github.com/WillWroble/MageZero
```

### **5.2 Create Virtual Environment**

use conda or your ML package manager of choice
```
cd MageZero
python -m venv .venv
.venv/bin/activate
pip install -r requirements.txt

```
you'll probably need to do the torch install yourself for your CUDA/hardware see `https://developer.nvidia.com/cuda-downloads`

### **5.3 GPU Check**

```
python -c "import torch; print(torch.cuda.is_available())"
```

---

## **6. Training a Deck Model**

### **6.1 Organize the Data**

When adding your XMage data make sure to add it according to this file structure. 
`data/<DECK-NAME>/ver<VERSION-NUMBER>/training/<YOUR-DATA-FILES>.hdf5`
(you might need to create these folders yourself) example:

```
data/
    UWTempo/
        ver7/
            training/
                *.hdf5
```
it's also recommended to add a separate .hdf5 data file under `../testing/` to eval your model's performance (value loss should be < 0.04) 
### **6.2 Configure Training Options**

Navigate to the editable constants at the top of `train.py`:

* `DECK_NAME`
* `VER_NUMBER`
* `EPOCH_COUNT`
* `USE_PREVIOUS_MODEL`

* change DECK_NAME and VER_NUMBER to correspond to where you put your training data in the file structure.
* change EPOCH_COUNT and USE_PREVIOUS_MODEL as you see fit (60 recommended for starting with random weights, 10 for model reuse)
* also optionally change `PRIORITY_A_MAX` `PRIORITY_B_MAX` `TARGETS_MAX` with the max index of the action mapping generated in XMage (will be printed at the
start of each RL run) this slightly improves network performance by properly normalizing loss


### **6.3 Run Training**

```
python train.py
```


* Starts reading shards
* does some preprocessing to automatically save and generate an ignore list/mask over feature indices
* Reports loss on test set (if provided) for value + policy heads per epoch
* automatically saves final model and ignore list to `models/<DECK-NAME>/ver<VER-NUM>/`
* old models are automatically overridden, if you want to be able to go back to a prev model bump version number

---

## **7. Running the Inference Server**

### **7.1 Start the server**

all fields you set in `train.py` are reused here 

run:
```
waitress-serve --host=127.0.0.1 --port=50052 --threads=8 server:app
```
(port is automatically set to 50052 in XMage, use 50053 for an opponent model if doing RL vs RL)

### **7.2 XMage → Python communication**

Explained briefly:

* XMage calls HTTP endpoint
* Server returns policy + value
* GPU needed to run inference hundreds of times per second during online simulations
* should see ~50 Evals/sec per thread on average.

once the server is running, start your RLvsMinimax run in XMage like before, it should find the endpoint and use the 
network now (performance might be slightly worse after one gen, perf should go up a lot in gen 2, 3 and 4 using NUM_GAMES = 1000)

* once the new data is generated *add* the new .hdf5 to the directory with the old .hdf5s. don't replace old training data 
unless you have 5000+ games of newer data, and you're reusing the network. 
* you can also run `dataset_stats.py` if you want to see histograms/plots for the datasets generated by XMage.
* train the new model with `train.py` like before
* close the old server and rerun it with new model (automatically updated by train.py)
* start gen 2 in XMage and continue the self play loop. It is highly recommended that you monitor network and agent performance
as you go, tweaking/annealing values as you see fit. Decks in MTG vary a lot, and this is far from a complete system. Enjoy :)


---



## **8. Evaluating**


### **8.1 Output**

* 'Composite children': the MCTS visit distributions for possible actions at current decision point.
should be flat at first and become more spikey later on.
* 'Actions:' the visit counts and MCTS pooled Q value (score) from (-1,1) for each possible action
example: `INFO  2025-11-23 19:03:49,433 PRECOMBAT_MAIN0 actions: [Play Adarkar Wastes score: 0.960 count: 42] [Play Meticulous Archive score: 0.953 count: 44] [Play Island score: 0.972 count: 67] [Pass score: 0.985 count: 146]  =>[pool-10-thread-1] MCTSNode.bestChild 
`
* evaluations/sec corresponds to how many network inferences are happening per sec for that thread
example: `INFO  2025-11-23 19:07:36,018 Total: simulated 17443 evaluations in 306.3927939919999 seconds - Average: 56.930190076387525 =>[pool-10-thread-3] ComputerPlayerMCTS2.applyMCTS 
` means you are exploring and evaluating 56 possible futures/sec per thread. if lower than 40 consider lowering `MAX_CONCURRENT_GAMES`
* WR data is saved to `Mage.Tests/train_results.txt` you can also view it live with a grep filter for 'WR:' on console
### **8.2 Best Practices**

* Use ≥200 games per matchup
* Compare WR across generations
* Never trust <100 games for WR

---

## **9. Troubleshooting / Common Issues**

TODO

---
## **10. Limitations**

* Many niche abilities in MTG (enter the dungeon, controlling opponents turn, pre game decisions)
These should work in XMage but will be invisible to the RL agent. (If you are a contributor looking to change that look at `StateEncoder.java`)
* Mulligan and side boarding turned off for simplicity right now.
* Hidden info, turned off by default for less noisy testing and model evaluations. Both players play with perfect info.
* Human vs AI support. currently being worked on.
* Manual mana payment. manual mana payment explodes the branching factor of MTG, for scalability we use XMages 
built in auto tapper to handle mana payment. (non mana costs are still simulated out in the MCTS tree). This also means
activating mana sources with nothing to pay for isn't available to the agent (can be relevant in niche situations)
* Niche micro decisions (i.e. ordering triggered abilities, choosing replacement effects, assigning damage) these aren't
conceptually a problem but just don't come up enough to justify implementation yet.

---


## **11. Roadmap & How to Get Involved**


* Discord: `inkling_6`
* Email: `willwroble@gmail.com`
* Contributions wanted: Java performance, network testing, new decks, XMage bug hunting

---


