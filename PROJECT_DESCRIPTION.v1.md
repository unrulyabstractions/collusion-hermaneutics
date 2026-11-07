#Title
Heurmaneutics for Collaboration Monitoring

#Goal
Cheap monitoring of collaboration to prevent unexpected collusion

#Approach
studying a closed-weight model (the author model) through the internals of an open-weight substitute (the reader model)
for collaboration 

#Background
1. https://www.lesswrong.com/posts/CZLhfNGaDqtYxvGJD/model-hermeneutics-monitoring-closed-weight-models-with-open
2. https://aryafayyazi.github.io/GreyBox/
3. https://arxiv.org/pdf/2606.32008
4. https://arxiv.org/pdf/2606.32008


#Needs
- Transcript data to build probes
- Labeled data for validation

#Probes Types
- A: One probe per author
- B: One probe using many authors
	* Trained in all, validated in excluded samples
	* Trained in some, validated in unseen authors


#Probes Configs
- Input: Residual stream
- Position
	* "<think>" + "</think>" tokens
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


#Possible data sources
- self-generate with cheatbench + orbit in own environments
- https://huggingface.co/datasets/Goodfire/reward-hack-data for evaluation?

#Our Environments
- Sizes:
	* 2 agent 
	* Multi-model swarm
	* Same-model swarm


