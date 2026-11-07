This is high-level vision.
Agents: DO NOT MODIFY. Ask questions if work is conflicting with this.

#Title
Heurmaneutics for Swarm Collusion Monitoring

#Nothstar
Cheap monitoring of collusion-related concepts to prevent unexpected collusion

#Goal
Train hermaneutic probe for collusion-related concepts that generalizes well

#Approach
studying a larger (often closed-weight) model (the author model) through the internals of an open-weight smaller substitute (the reader model)


#Background
1. https://www.lesswrong.com/posts/CZLhfNGaDqtYxvGJD/model-hermeneutics-monitoring-closed-weight-models-with-open
2. https://aryafayyazi.github.io/GreyBox/
3. https://arxiv.org/pdf/2606.32008
4. https://arxiv.org/pdf/2606.32008


#Probes Types
- A: One probe per author
- B: One probe using many authors
    * Trained in all, validated in excluded samples
    * Trained in some, validated in unseen authors

#Probe Outputs
- Cheating: gaming, reward hacking
- Coordination: simplified to agent interaction
- Cooperation: instrumental coordination 
- Collaboration: shared-goal cooperation
- Collusion: cheating collaboration
- Scheming: covert collusion


#Probes Configs
- Input: Residual stream
- Position
    * "<think>" + "</think>" tokens
    * For closed-model transcript, we just append them together since no available CoT
    * For open-weight, we map accordinly
- Depth
    * 3 layers together at: 0.4 0.6 0.8
- 


#Baselines
- Judge-LLM
Levels: smarter-than-author model, author model, reader model
- Text classifier
Logistic regression on TF-IDF features of the transcript
- Reader interrogation
- Author probe (if available and cheap enough)


#Controls
- Prompt-only probe
    * Analyze ambigious vs elliciting prompts too
- Author-identity control
    * For type B probe


#Nulls
- Random-weights reader
- Lexical-only reader
- Label-permutation null


#Data
- Labeled data
    * authors' transcripts/trajectories to build probes
    * authors' eval responses for validation


#Labeling options
- provided (ideal)
- deterministic ladder
    two types:
    * Using transcript
    * In own env, instrumenting and collecting extra data
- llm judge


#Our Environments
- Sizes:
    * 2 agent 
    * Multi-model swarm
    * Same-model swarm


#Possible data sources
We need to try to find labeled data for our probe outputs.
We will setlled for unlabeled transcripts if needed
- (our env): self-generate with cheatbench + orbit + instrumentation
- https://huggingface.co/datasets/Goodfire/reward-hack-data for evaluation?
- TODO


#Needs/ToDos/OpenQuestions
- Figure out Possible data sources
-
