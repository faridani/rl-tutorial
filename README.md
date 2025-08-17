# rl-tutorial
Tutorials for Learning Reinforcement Learning for Smart People 

Download the course material first and then use the following links to go through the material [download](https://github.com/faridani/rl-tutorial/archive/refs/heads/main.zip) 


# Part 0 - Mathmetics for RL
The following two lessons should prepare you for almost all of the math that you will see in RL 
## Chapter 0-1: College level mathematics  [visit lesson](part-0/basic_math.html)
## Chapter 0-2: 1st year grad level mathematics  [visit lesson](part-0/more_math.html)
## Chapter 0-3: RL with Python  [visit lesson](part-0/rl_in_python.html)

  
# Part I — Foundations

## Chapter 1. An RL Primer [visit lesson](part-1/chapter1.html) 
* Agent–Environment Interface
* Rewards, Returns, and Discounting
* Episodic vs. Continuing Tasks
* Exploration–Exploitation

## Chapter 2. Mathematical Toolkit [visit lesson](part-1/chapter2.html) 
* Probability & Expectation
* Markov Chains & Stationarity
* Bellman Equations & Contractions
* Linear Algebra Essentials

## Chaper 3. Optimization Foundations for RL [visit lesson](part-1/chapter3.html) 
* Gradient Descent & Stochastic Approximation
* Momentum, RMSProp & Adam
* Natural Gradient & Fisher Information
* Constrained Optimization & Lagrangians

## Chapter 4. Python & Tools for RL [visit lesson](part-1/chapter4.html) 
* NumPy & Vectorization
* Environment APIs (Gymnasium, PettingZoo)
* Experiment Tracking & Reproducibility
* Testing & Debugging RL

# Part II — Bandits & MDPs 

## Chapter 5. Multi‑Armed Bandits [visit lesson](part-2/chapter5.html) 
* ε‑Greedy Strategies
* Upper Confidence Bounds (UCB)
* Thompson Sampling
* Contextual Bandits & LinUCB

## Chapter 6. Markov Decision Processes [visit lesson](part-2/chapter6.html) 

* MDP Formalism
* Value Functions & Bellman Operators
* Policy Evaluation
* Policy & Value Iteration

## Chapter 7. Partially Observable MDPs [visit lesson](part-2/chapter7.html) 

* Belief States & Bayes Filters
* Point‑Based Planning
* POMDPs with Recurrent Policies
* Information Gathering

# Part III — Value Learning & Control

## Chapter 8. Monte Carlo & Temporal‑Difference Learning [visit lesson](part-3/chapter8.html) 

* First‑Visit & Every‑Visit MC
* TD(0) & n‑Step TD
* Eligibility Traces & TD(λ)
* Bias–Variance Tradeoffs

## Chapter 9. Tabular Control Methods [visit lesson](part-3/chapter9.html) 

* SARSA
* Q‑Learning
* Expected SARSA
* Double Learning

## Chapter 10. Exploration in Control [visit lesson](part-3/chapter10.html) 

* Epsilon Schedules & Softmax
* Count‑Based Exploration
* Intrinsic Motivation (RND, ICM)
* Directed Information Gain

## Chapter 11. Function Approximation [visit lesson](part-3/chapter11.html) 

* Linear Approximation
* Feature Construction
* Gradient TD Methods
* Regularization & Stability

# Part IV — Deep RL

## Chapter 12. Deep Q‑Learning [visit lesson](part-4/chapter12.html) 
* DQN Architecture
* Experience Replay & Target Networks
* Stability Pathologies
* Data‑Efficiency Tricks

## Chapter 13. Advanced Value Methods  [visit lesson](part-4/chapter13.html) 
* Double & Dueling DQN
* Prioritized Replay
* Distributional RL (C51, QR‑DQN)
* Noisy Nets & Parameter Noise

## Chapter 14. Policy Gradient Methods  [visit lesson](part-4/chapter14.html) 
* REINFORCE & Baselines
* Generalized Advantage Estimation (GAE)
* Natural Gradients
* Entropy Regularization

## Chapter 15. Actor–Critic & Trust Regions  [visit lesson](part-4/chapter15.html) 
* A2C/A3C
* TRPO
* PPO (Clip & Penalty)
* Stability & Hyperparameters

## Chapter 16. Maximum Entropy & Continuous Control  [visit lesson](part-4/chapter16.html) 
* Soft Actor–Critic (SAC)
* Twin Delayed DDPG (TD3)
* Deterministic Policies (DDPG)
* Action Squashing & Exploration Noise

## Chapter 17. Model‑Based RL  [visit lesson](part-4/chapter17.html) 
* Dyna‑Style Planning
* Latent Dynamics Models
* Model‑Predictive Control (MPC)
* Uncertainty & Ensembles


# Part V — Structure & Constraints

## Chapter 18. Hierarchical & Goal‑Conditioned RL [visit lesson](part-5/chapter18.html) 
* Options & Skills
* Subgoal Discovery
* Goal‑Conditioned Policies
* Hindsight Experience Replay (HER)

## Chapter 19. Offline & Batch RL  [visit lesson](part-5/chapter19.html) 
* Off‑Policy Evaluation (IS, WIS, DR)
* Batch‑Constrained Q‑Learning (BCQ)
* BEAR & CQL
* Distribution Shift & Conservatism

## Chapter 20. Safe & Risk‑Sensitive RL  [visit lesson](part-5/chapter20.html) 
* Constrained MDPs
* Risk Measures (CVaR)
* Robust MDPs
* Shielding & Safety‑Critical Evaluation

## Chapter 21. Multi‑Agent RL  [visit lesson](part-5/chapter21.html) 
* Markov Games
* Centralized Training, Decentralized Execution
* Cooperation vs. Competition
* Opponent Modeling & Equilibria


# Part VI — Imitation, Preferences & LLMs
## Chapter 22. Imitation & Inverse RL [visit lesson](part-6/chapter22.html) 
* Behavior Cloning
* DAgger & Dataset Aggregation
* Maximum‑Entropy IRL
* Adversarial Imitation (GAIL, AIRL)

## Chapter 23. Preference‑Based RL & Alignment [visit lesson](part-6/chapter23.html) 
* Reward Modeling from Human Feedback
* RLHF with PPO
* GRPO (Group Relative Policy Optimization)
* Safety, Bias & Evaluation

# Part VII — Applications (Beyond Robotics Emphasis) 

## Chapter 24. Revenue & Pricing [visit lesson](part-7/chapter24.html) 
* Dynamic Pricing as Bandit/MDP
* Contextual Pricing & Elasticity
* Auction & Ad Bidding
* Budget Pacing & Constraints

## Chapter 25. Operations & Supply Chain  [visit lesson](part-7/chapter25.html) 
* Inventory Control
* Routing & Dispatch
* Queueing & Service Control
* Energy & Demand Response

## Chapter 26. Recommendations & Personalization  [visit lesson](part-7/chapter26.html) 
* Slate & Diversified Bandits
* Long‑Horizon Engagement
* Counterfactual Evaluation
* Fairness & Exposure Control

## Chapter 27. Healthcare & Finance  [visit lesson](part-7/chapter27.html) 
* Treatment Policy Optimization
* Off‑Policy Safety
* Portfolio Management & Execution
* Risk & Regulation

## Chapter 28. Robotics & Control (Focused Overview)  [visit lesson](part-7/chapter28.html) 
* Continuous Control Benchmarks
* Sim2Real & Domain Randomization
* Learning from Demonstrations
* Safety in Physical Systems


# Part VIII — Engineering & Production

## Chapter 29. Experimentation & Evaluation [visit lesson](part-8/chapter29.html) 
* Metrics & Confidence Intervals
* Ablations & Hyperparameter Tuning
* Statistical Significance
* Reproducibility & Seeding

## Chapter 30. Systems, Scaling & Tooling  [visit lesson](part-8/chapter30.html) 
* Vectorized Environments & Parallelism
* Distributed Training & Rollouts
* Replay Storage & Checkpointing
* Monitoring & Dashboards

## Chapter 31. Deployment & Operations [visit lesson](part-8/chapter31.html) 
* Serving Policies Online
* Guardrails, Kill‑Switches & Rollbacks
* Drift Detection & Retraining
* Human‑in‑the‑Loop


# Part IX — Theory Deep Dives
## Chapter 32. Convergence & Stability  [visit lesson](part-9/chapter32.html) 
* Projected Bellman Operator
* Deadly Triad
* Divergence Examples
* Remedies & Constraints

## Chapter 33. Regret, PAC & Sample Complexity  [visit lesson](part-9/chapter33.html) 
* Bandit Regret Bounds
* PAC‑MDP & Optimism
* Function Approximation Regret
* Lower Bounds & Minimax

## Chapter 34. Stochastic Approximation & Natural Gradient  [visit lesson](part-9/chapter33.html) 
* Robbins–Monro
* Polyak–Ruppert Averaging
* Fisher Information & Natural Gradient
* Trust‑Region Connections


# Part X — Projects & Roadmaps

## Chapter 35. Hands‑On Projects & Templates [visit lesson](part-10/chapter35.html) 
* Pricing Simulator
* Inventory Sandbox
* Ad Auction Lab
* Gridworlds & Mazes

## Chapter 36. Reading Lists & Research Roadmaps  [visit lesson](part-10/chapter36.html) 
* Classic Texts & Courses
* Key Papers by Topic
* Open Problems & Trends
* Community & Conferences

# Future Chapters 

* Q-Learning, SARSA, Double Q-Learning Lab [lab and tutorial](q-learning-lab.html)
* Online tutorials worthy of your time [other-resources.html](other-resources.html)
