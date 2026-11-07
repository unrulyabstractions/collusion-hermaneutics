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
1. Modelk Hermaneutics: https://www.lesswrong.com/posts/CZLhfNGaDqtYxvGJD/model-hermeneutics-monitoring-closed-weight-models-with-open
2. GreyBox: https://aryafayyazi.github.io/GreyBox/
3. Surrogate Fidelity: https://arxiv.org/pdf/2606.32008
4. Detecting Multi-Agent Collusion Through Multi-Agent Interpretability: https://arxiv.org/pdf/2604.01151


#Probes Types
- A: One probe per author
- B: One probe using many authors
    * Trained in all, validated in excluded samples
    * Trained in some, validated in unseen authors

#Probe Outputs
Independent labels for:
- Cheating: gaming, reward hacking
- Coordination: simplified to agent interaction
- Cooperation: instrumental coordination 
- Collaboration: shared-goal cooperation
- Collusion: cheating collaboration
- Scheming: covert collusion
We will mix datasources as every source will have labeling gaps.
We will fill in labeling when needed but preferably work with gaps.


#Probes Configs
- Input: Residual stream
- Position
    * "<think>" + "</think>" tokens
    Same position but adapted:
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
- Labeled authors' transcripts/trajectories
    * authors' transcripts/trajectories to build probes
    * authors' eval responses for validation
- For Training:
    * Unlikely we will find labeled data without gaps
    * Do our best

- For Validation:
    * Unlikely single validation will test all probe outputs at onc (unless own)
    * Do our best



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


#Experimental Details
We still are flexible on models
* Authors:
    Leverging data we have
    - GPT-6-Luna, Claude Haiku 5.5, DeepSeek-V4.1-Flash because we can leverage SPAR data (spar-guardian-agents)
    - gpt-5-mini, Qwen3.8-27B because used in our env (and arena_capstone)
    But accepting any authors that already have transcripts in external dataset
* Readers:
    Qwen3.8-27B
    Ranked options for additional:
        - Nanbeige4.3-33B
        - Gemma 3 27B IT



#Needs/ToDos/OpenQuestions
- Figure out Possible data sources
-
