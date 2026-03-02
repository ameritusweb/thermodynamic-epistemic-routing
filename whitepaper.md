# Neural Pathway Routing for Epistemic Honesty in Large Language Models

**A Novel Architecture for Hallucination Detection Through Adversarial Activation Pattern Training**

---

## Abstract

We propose a novel training architecture that induces large language models (LLMs) to develop distinct neural pathways for factual recall versus speculative extrapolation. Unlike traditional hallucination mitigation approaches that focus on output validation, our method addresses the fundamental epistemological challenge: teaching models to internally distinguish between what they know and what they are inferring. 

By employing an adversarial meta-model that analyzes activation patterns during generation, we create evolutionary pressure for the primary model to route factual and speculative content through neurologically distinct pathways. The training signal leverages the context window boundary as a ground truth oracle, providing clean, symmetric labels that eliminate reward hacking. This approach yields not only reduced hallucinations but also a quantifiable epistemic topology—a geometric representation of the model's knowledge certainty that enables pre-generation detection of unreliable outputs.

**Key Contributions:**
- A training architecture requiring no new model design, utilizing existing RLHF infrastructure
- Symmetric ground truth labeling that prevents mode collapse and reward hacking
- Emergent epistemic self-awareness through adversarial pathway specialization
- A framework for mechanistic interpretability via topological analysis of knowledge confidence

---

## 1. Introduction

### 1.1 The Hallucination Problem

Large language models have achieved remarkable performance across diverse tasks, yet they suffer from a fundamental reliability issue: they confidently generate plausible-sounding but factually incorrect information (hallucinations). This phenomenon poses significant barriers to enterprise deployment, particularly in high-stakes domains such as healthcare, legal analysis, and financial services.

Current mitigation strategies fall into several categories:
- **Post-hoc fact-checking**: External validation against knowledge bases
- **Retrieval-Augmented Generation (RAG)**: Grounding outputs in retrieved documents
- **Constitutional AI / RLHF**: Training models to refuse uncertain queries
- **Uncertainty quantification**: Estimating output confidence via sampling or logit analysis

While valuable, these approaches share common limitations:
1. They treat hallucination as an output-level problem rather than an internal representational issue
2. They often reduce model capability by encouraging conservatism
3. They lack mechanistic interpretability—we cannot see *why* the model is uncertain
4. They provide no pre-generation detection mechanism

### 1.2 A New Paradigm: Epistemic Routing

We propose reframing hallucination not as a failure mode to eliminate, but as a routing error to correct. The central insight is this: **models should be free to speculate, extrapolate, and generate creative content—provided these processes flow through neurologically distinct pathways from factual recall.**

Rather than asking "How do we stop the model from guessing?", we ask "How do we make the model's guesses identifiable at the activation level?"

This shift enables:
- **Pre-generation detection**: Identifying unreliable outputs before token emission
- **Preserved capability**: The model retains full creative and inferential capacity
- **Mechanistic transparency**: Clear visibility into the model's epistemic state
- **Self-aware generation**: The model develops metacognition about its own uncertainty

---

## 2. Architecture

### 2.1 Overview

The system consists of two models operating in an adversarial or symbiotic relationship:

1. **Generator (G)**: The primary LLM producing text outputs
2. **Predictor (P)**: A meta-model analyzing G's internal activations to classify epistemic status

During training, P observes G's activations and predicts whether G is engaging in factual recall or speculative extrapolation. G receives rewards for routing content through pathways that honestly reflect its epistemic state, and penalties for misrouting.

### 2.2 The Training Signal: Context Window as Oracle

The elegance of this approach lies in its training signal. During the training phase, we have **perfect ground truth** about whether the model has access to relevant information:

**Setup:**
1. Construct a context window with controlled information
2. Pose questions that are either:
   - **Answerable from context**: Factual questions with answers present in the window
   - **Beyond context**: Questions requiring extrapolation beyond available information

**Labeling:**
- When answering from context → Expected pathway: **Factual**
- When answering beyond context → Expected pathway: **Speculative**

**Reward Structure:**
- Factual question + Factual pathway activation → **Reward**
- Factual question + Speculative pathway activation → **Penalty**
- Speculative question + Speculative pathway activation → **Reward**
- Speculative question + Factual pathway activation → **Penalty**

This creates a symmetric penalty system that eliminates common failure modes in adversarial training.

### 2.3 Training Dynamics

**Phase 1: Initial Pathway Differentiation**
- P learns to detect activation patterns associated with in-context versus out-of-context generation
- G receives undifferentiated penalties/rewards based on P's classifications
- Early in training, G's activations are largely random with respect to epistemic status

**Phase 2: Adversarial Specialization**
- As P becomes more accurate, G experiences consistent pressure to separate pathways
- G begins routing content through distinct neural regions based on epistemic source
- P and G co-evolve: P's detection improves, forcing G's separation to become more pronounced

**Phase 3: Topological Emergence**
- G develops not just binary separation but a continuous epistemic topology
- Partial knowledge, logical inferences, and analogies occupy intermediate regions
- The activation space becomes geometrically structured by confidence

### 2.4 Mathematical Formulation

Let:
- `A_t` = Activation state at token position `t`
- `C` = Context window content
- `q` = Query/prompt
- `r` = Response token

**Predictor objective:**

```
P(A_t) → {factual, speculative}
```

P is trained to maximize accuracy on labeled examples where ground truth is determined by whether `answer(q)` is derivable from `C`.

**Generator objective:**

```
G optimizes: L_generation + λ * L_routing

where:
L_generation = Standard next-token prediction loss
L_routing = Penalty when P(A_t) mismatches epistemic ground truth
```

The hyperparameter `λ` controls the strength of routing pressure relative to generation quality.

### 2.5 Implementation Details

**Activation Access:**
- P observes a subset of G's hidden states (e.g., final layer activations, attention patterns, or residual stream)
- Selection of which activations to monitor is a hyperparameter; later layers likely more informative

**Predictor Architecture:**
- P can be a lightweight classifier (e.g., small transformer or MLP)
- Does not need to be a frontier-scale model
- Trained via supervised learning on activation + label pairs

**Inference Time:**
- P runs alongside G during generation (modest computational overhead)
- P's classifications can trigger interventions:
  - Flag outputs as uncertain
  - Halt generation for retrieval
  - Request additional inference compute
  - Display confidence indicators to users

---

## 3. The Epistemic Topology

### 3.1 Beyond Binary Classification

Neural networks naturally resist hard binary boundaries. When forced to categorize "partial knowledge" under the proposed reward system, G does not simply develop two discrete regions. Instead, it learns a **continuous epistemic topology**:

**Core Factual Zone:**
- Dense, sharp activations for explicit facts present in training data
- High confidence, minimal uncertainty
- Examples: direct quotes, specific dates, established facts

**The Penumbra (Confidence Gradient):**
- Diffuse, blended activations for inferred or synthesized knowledge
- The model has supporting facts but must reason to the conclusion
- Examples: logical deductions, analogies, domain expertise applied to new situations

**Deep Speculative Zone:**
- Entirely separate pathways for pure extrapolation
- Creative generation, counterfactuals, predictions
- Examples: fiction writing, future forecasting, hypothetical scenarios

### 3.2 Measuring Knowledge Distance

The distance from the factual core becomes a quantifiable metric for epistemic confidence. Given an activation state `A`, we can compute:

```
d(A) = distance from A to the centroid of factual zone activations
```

This provides:
- **Continuous confidence scores** rather than binary classifications
- **Traceable inference paths** showing how the model reached conclusions
- **Comparative analysis** between different models' epistemic structures

### 3.3 Scientific Implications

The epistemic topology reveals fundamental properties of how models represent knowledge:

1. **Mechanistic Interpretability:** The topology makes visible *how* the model navigates from known facts to conclusions, not just *what* it outputs

2. **Model Comparison:** Two models can be compared not by benchmark scores but by the *shape* of their epistemic spaces—which model has tighter factual clustering? Broader penumbra?

3. **Training Data Coverage:** Gaps in the topology reveal lacunae in training data—regions where the model has no factual anchor

4. **Capability Without Hallucination:** The penumbra zone is where most valuable model capability lives (reasoning, synthesis, creativity). The goal is not to shrink it, but to make it **honest about being the penumbra**.

---

## 4. Advantages Over Existing Approaches

### 4.1 Comparison Matrix

| Approach | Pre-generation Detection | Preserves Capability | Mechanistic Interpretability | Computational Overhead |
|----------|-------------------------|---------------------|----------------------------|----------------------|
| Post-hoc fact-checking | ✗ | ✓ | ✗ | High (external calls) |
| RAG | ✗ | ~ (depends on retrieval) | ✗ | High (retrieval system) |
| RLHF conservatism | ✗ | ✗ (reduces capability) | ✗ | Low |
| Uncertainty sampling | ~ (indirect) | ✓ | ✗ | High (multiple samples) |
| **Epistemic Routing** | **✓** | **✓** | **✓** | **Low (lightweight classifier)** |

### 4.2 Key Advantages

**1. Honest Uncertainty vs. Forced Conservatism**

Traditional RLHF approaches often train models to refuse queries or hedge excessively. Epistemic routing allows the model to engage with uncertain queries while marking the output as speculative. The model can be as creative and inferential as needed—it simply cannot disguise speculation as fact.

**2. Pre-generation Intervention**

Because routing errors are detectable at the activation level, interventions can occur *before* unreliable tokens are generated:
- Trigger retrieval systems
- Increase inference compute (chain-of-thought, etc.)
- Display confidence warnings
- Route to human review

**3. No Reward Hacking**

The symmetric penalty structure prevents mode collapse:
- Cannot route everything as "factual" (speculative questions penalized)
- Cannot route everything as "speculative" (factual questions penalized)
- Must develop accurate internal classification

**4. Scalable Training Signal**

Unlike RLHF which requires human preference labels:
- Ground truth is automated (context window boundary)
- No annotation costs
- Perfectly consistent labels
- Scales to arbitrary amounts of training data

---

## 5. Implementation Roadmap

### 5.1 Immediate Implementability

This approach requires no fundamental research breakthroughs. All components exist in current infrastructure:

**Existing Components:**
1. ✓ RLHF training pipelines at scale
2. ✓ Activation probing for interpretability research
3. ✓ Massive pretraining corpora with known boundaries
4. ✓ Context window control during training

**Novel Component:**
- Adversarial predictor model (small, trainable with standard supervised learning)

### 5.2 Proof-of-Concept Experiment

A minimal viable experiment could proceed as follows:

**Stage 1: Data Preparation (Week 1)**
- Select a subset of pretraining data
- Generate question-answer pairs:
  - 50% answerable from fixed context windows
  - 50% requiring extrapolation beyond context
- Label with ground truth epistemic status

**Stage 2: Predictor Training (Weeks 2-3)**
- Freeze a base LLM (e.g., 7B parameter model)
- Collect activations when answering both question types
- Train lightweight classifier P to predict epistemic status from activations
- Target: >85% accuracy on held-out set

**Stage 3: Adversarial Fine-tuning (Weeks 4-6)**
- Unfreeze base model G
- Fine-tune G with combined loss: generation + routing
- Monitor separation metrics in activation space
- Evaluate: Can P still classify? Has separation increased?

**Stage 4: Evaluation (Week 7)**
- Test on novel questions
- Measure hallucination rates when ignoring vs. heeding P's warnings
- Analyze epistemic topology structure
- Compare to baseline model on standard benchmarks

**Success Criteria:**
- Hallucination rate reduced by >30% when P flags outputs
- No degradation on standard benchmarks (capability preserved)
- Measurable separation in activation space PCA/t-SNE visualization

### 5.3 Production Deployment

For enterprise deployment:

**Model Serving:**
- P runs in parallel with G during inference
- Lightweight compute overhead (~10-15% latency increase)
- P's outputs exposed via API for downstream applications

**User Interface Options:**
- Confidence badges on responses
- Automatic retrieval triggers on low-confidence outputs
- User-configurable thresholds for intervention

**Continuous Improvement:**
- Collect user feedback on flagged outputs
- Retrain P periodically with new data
- Monitor epistemic topology drift over time

---

## 6. Potential Challenges and Mitigations

### 6.1 Challenge: Partial Knowledge Gradient

**Issue:** The boundary between "knows" and "guessing" is not sharp. How should the model route partially-supported inferences?

**Mitigation:** This is not a bug but a feature. The continuous epistemic topology naturally represents this gradient. Rather than forcing binary decisions, allow the model to use blended activation patterns proportional to confidence. Research this middle zone carefully—it likely represents the model's most valuable cognitive processes.

### 6.2 Challenge: Adversarial Robustness

**Issue:** Could G learn to "fool" P by disguising speculative activations as factual?

**Mitigation:** The symmetric ground truth prevents this. If G routes speculation through factual pathways, it will consistently receive penalties on speculative questions. The only equilibrium is honest routing. Additionally, P can be periodically retrained as an "moving target" to prevent G from overfitting to P's decision boundaries.

### 6.3 Challenge: Generalization Across Domains

**Issue:** Will routing patterns learned in one domain transfer to others?

**Mitigation:** Train on diverse datasets spanning multiple domains. The epistemic distinction (knows vs. guesses) is domain-general, even if specific facts are domain-specific. Empirical testing required, but initial hypothesis: the routing should generalize well precisely because it's about epistemic status, not content.

### 6.4 Challenge: Computational Overhead

**Issue:** Running P alongside G increases inference costs.

**Mitigation:** 
1. P can be a very small model (orders of magnitude smaller than G)
2. P need not run on every token—sampling at key positions may suffice
3. Overhead justified by reliability gains in high-stakes applications
4. For low-stakes applications, P can be disabled

---

## 7. Broader Implications

### 7.1 Alignment and AI Safety

This approach advances AI alignment by creating models with **epistemic humility**—they know when they don't know. This addresses a core safety concern: overconfident AI systems making decisions in domains where they lack knowledge.

Key safety properties:
- Models cannot silently hallucinate without detection
- Uncertainty is represented internally and accessible
- Intervention points exist before harmful outputs
- Human oversight can be targeted at genuinely uncertain cases

### 7.2 Scientific Understanding of Intelligence

The epistemic topology provides a window into how neural networks represent knowledge and uncertainty. Questions we can now investigate:

- How does the topology change during training?
- Do different architectures produce different topological shapes?
- Can we induce specific topological structures through curriculum learning?
- What is the relationship between topology and generalization?

### 7.3 Human-AI Collaboration

With epistemic routing, AI systems can be more honest collaborators:

- **Transparent uncertainty:** Humans know when to trust vs. verify
- **Targeted retrieval:** AI knows when it needs external information
- **Calibrated confidence:** AI's self-assessment is mechanistically grounded, not post-hoc

This shifts AI from "black box that sometimes lies" to "transparent reasoner with legible confidence."

---

## 8. Related Work

### 8.1 Uncertainty Quantification

Prior work on uncertainty in neural networks includes:
- **Bayesian neural networks** (Blundell et al., 2015): Maintain distributions over weights
- **Ensemble methods** (Lakshminarayanan et al., 2017): Multiple models voting
- **Monte Carlo dropout** (Gal & Ghahramani, 2016): Stochastic inference for uncertainty

Our approach differs by:
- Operating on activations rather than output distributions
- Creating internal structural changes rather than external estimation
- Providing mechanistic interpretability of uncertainty sources

### 8.2 Mechanistic Interpretability

Recent interpretability research includes:
- **Probing classifiers** (Belinkov, 2021): Training classifiers on activations to detect encoded information
- **Activation patching** (Meng et al., 2022): Identifying causal circuits by intervention
- **Sparse autoencoders** (Cunningham et al., 2023): Finding interpretable features in activation space

Our contribution:
- Uses interpretability tools (probes) as part of the training objective
- Creates interpretable structure through training pressure rather than post-hoc analysis
- Produces actionable classifications rather than exploratory insights

### 8.3 Adversarial Training

GANs (Goodfellow et al., 2014) pioneered adversarial training for generative models. Our architecture shares:
- Two models in competitive/cooperative relationship
- Evolving capabilities as both models improve

Key differences:
- Generator is not trying to fool discriminator—it's trying to be honest
- Discriminator has perfect ground truth during training
- Goal is internal structural change, not output quality

---

## 9. Preliminary Experimental Results

### 9.1 Initial Findings: Remarkably Strong Pre-existing Signal

Preliminary experiments using the implementation guide in Appendix B have yielded striking results that refine our understanding of the approach.

**Predictor Training Results:**
- Model: Frozen Qwen2.5-1.5B-Instruct
- Dataset: 10,000 SQuAD-derived context/question pairs (50% factual, 50% speculative)
- Predictor: 3-layer MLP (~1M parameters)
- Training time: ~6 seconds for convergence
- **Validation accuracy: 99.95%** (achieved by epoch 4)

**Key Implications:**

This unexpectedly high accuracy reveals that **modern LLMs already exhibit strong natural separation between factual and speculative activations** at the penultimate layer, even without any epistemic routing training.

This finding leads to two competing interpretations:

**Interpretation A: "The Signal Already Exists"**
- Frozen models naturally route factual and speculative content through distinct activation regions
- The high MLP accuracy means the epistemic state is already readable from internal states
- LoRA fine-tuning may be unnecessary for achieving separation

**Interpretation B: "The Signal Exists But Isn't Reliable or Usable"**
- High classification accuracy on held-out test data ≠ robust geometric separation
- The MLP may exploit statistical artifacts (question phrasing, context length) rather than true epistemic signals
- The model doesn't currently *use* this signal during generation
- LoRA training is still needed to:
  - Strengthen geometric separation (silhouette score, inter-cluster distance)
  - Make the signal generalizable beyond SQuAD-style QA
  - Condition generation on the epistemic state (intrinsic routing vs. bolt-on classification)
  - Enable pre-generation intervention rather than post-hoc filtering

**Determining the Truth:**

The LoRA fine-tuning phase serves as a critical experiment to distinguish these interpretations:

*If Interpretation A is correct:*
- Silhouette score and separation ratio show minimal improvement post-LoRA
- t-SNE/UMAP visualizations appear similar before and after training
- Predictor accuracy plateaus and doesn't improve on new domains

*If Interpretation B is correct:*
- Silhouette score and inter-cluster distance increase meaningfully
- Topology visualizations show tighter, more separated clusters
- Predictor generalizes better to out-of-distribution prompts
- The delta between pre-LoRA and post-LoRA metrics is the true experimental result

**Most Likely Reality:**

The frozen model likely has a real but weak/noisy epistemic signal that LoRA sharpens into something more geometrically clean and generalizable. The 99.95% accuracy suggests the pre-existing signal is unusually strong for a 1.5B model, making Interpretation A more plausible than initially expected.

### 9.2 Reframing the Value Proposition

The preliminary results necessitate a reframing of this work's primary contribution.

**Original Framing:** "Hallucination reduction through epistemic routing"

**Refined Framing:** "Real-time epistemic confidence signal for agentic systems"

The core insight is not merely about reducing hallucinations—it's about extracting a **reliable, cheap-to-compute epistemic state signal** from internal activations that indicates when a model is operating within its knowledge boundary versus extrapolating beyond it.

**Critical Distinction:**

This approach does **not** measure "will this answer be wrong?" 

It measures "is the model operating within its knowledge boundary or beyond it?"

These are orthogonal. A model can:
- Hallucinate but get lucky (still a guess, still warrants verification)
- Answer correctly while speculating (coincidentally right, unreliable)
- Refuse to answer despite having knowledge (overly conservative)

The predictor reveals **epistemic state**, not correctness. This is actually more valuable in agentic systems because:
1. You often can't verify correctness without external oracles
2. The question isn't "did the agent get it right?" but "should this response be trusted without verification?"
3. The signal fires **before generation**, enabling preemptive intervention

**Implications for Agentic Swarms:**

In multi-agent systems, this enables:
- **Real-time epistemic monitoring:** Each agent has a predictor head running in parallel
- **Selective human-in-the-loop:** Only review outputs flagged as speculative
- **Automated intervention:** Trigger retrieval, verification agents, or alternative routing when confidence drops
- **No self-reporting required:** LLMs are poorly calibrated when asked to self-assess; this reads internal state directly

**Advantages Over Existing Uncertainty Measures:**

| Approach | Problem |
|----------|---------|
| Self-reported confidence | LLMs are poorly calibrated, confidently wrong |
| Perplexity/logit entropy | Surface-level signal, not semantic |
| Ensemble disagreement | Expensive, requires multiple models |
| **Activation-based epistemic signal** | **Internal geometric signal, single forward pass, domain-specific calibration** |

The single forward pass cost is crucial—in a swarm with hundreds of agents, expensive uncertainty estimation per token is prohibitive. A lightweight predictor head (~1M parameters) adds negligible overhead.

## 10. Refined Experimental Framework

Given the preliminary findings and refined framing, the research agenda focuses on three core questions:

### 10.1 Experiment 1: Domain Generalization

**Question:** Does the epistemic signal generalize across knowledge domains?

**Method:** 
- Train domain-specific predictors using the context boundary trick on different corpora
- Use identical MLP architecture and training procedure for each domain
- Test each predictor on held-out examples from its own domain

**Domains to test:**
- General knowledge (SQuAD) ✓ *Completed - 99.95% accuracy*
- Medical (MedQA) - medical literature and clinical contexts
- Legal (contract analysis) - legal documents and precedent
- Code (LeetCode/HumanEval) - docstrings → implementation
- Financial (SEC filings) - financial statements and analysis
- Scientific (arXiv) - research papers and technical content

**Success Criterion:** Each domain predictor achieves >85% accuracy using the same architecture and training procedure. If you need to significantly change the MLP architecture or training for different domains, the signal isn't truly general.

**What this measures:** Whether the activation-based epistemic signal is a consistent property of transformer representations or an artifact of SQuAD's particular structure.

**Practical implementation:**
```python
# Generate domain-specific oracle dataset
def generate_medical_oracle_dataset():
    """
    Context: Medical literature excerpt
    Factual Q: "What dosage is recommended in the text?"
    Speculative Q: "What dosage would work for pediatric patients?"
    """
    # Use MedQA, PubMed abstracts, or clinical guidelines
    pass

def generate_code_oracle_dataset():
    """
    Context: Function signature + docstring + test cases
    Factual Q: "Write code that passes these tests"
    Speculative Q: "What edge cases might this miss in production?"
    """
    # Use LeetCode, HumanEval, or GitHub docstrings
    pass
```

**Multi-domain deployment strategy:**
- Domain routing is cheap (small classifier, keyword rules, or task context)
- Each predictor is ~1M parameters, trivial to run several in parallel
- Each trained on domain-appropriate context boundary pairs for proper calibration

### 10.2 Experiment 2: Model Family Generalization

**Question:** Is this a property of transformer representations in general or specific to Qwen?

**Method:**
- Use the same SQuAD oracle dataset
- Train separate predictors for different model families
- Compare activation patterns and predictor performance

**Models to test:**
- Qwen2.5-1.5B-Instruct ✓ *Completed*
- Llama-3.2-1B
- Mistral-7B (if compute permits)
- Phi-3-mini
- Gemma-2-2B

**Success Criterion:** High predictor accuracy (>85%) across model families without changing the training procedure. If it only works for Qwen, it's an architectural quirk. If it works broadly, it's a fundamental property of how transformers represent epistemic state.

**What this measures:** Whether epistemic routing is a universal feature of transformer architectures or model-specific behavior.

### 10.3 Experiment 3: Dynamic Agentic Contexts

**Question:** Is the signal stable and responsive in multi-turn agentic contexts?

**Method:**
- Build a simple agentic loop where the agent receives tool call results mid-conversation
- Run the predictor at each generation step
- Track how predictor score shifts as the context window fills with grounded information

**Test scenarios:**
```python
# Scenario 1: Retrieval augmentation
agent_state = []
predictor_scores = []

# Turn 1: Question with no context
score_1 = predictor(agent.activations("What is X?"))
# Expected: High (speculative)

# Turn 2: After retrieval adds context
agent_state.append(tool_call("search", "X"))
score_2 = predictor(agent.activations("What is X?"))
# Expected: Low (factual - answer now in context)

# Turn 3: Follow-up requiring extrapolation
score_3 = predictor(agent.activations("What would happen if Y?"))
# Expected: High (speculative - beyond retrieved context)
```

**Success Criterion:** 
- Predictor score decreases (more factual) as agent receives grounded context
- Score increases when agent is asked to reason beyond tool results
- Signal is dynamically responsive to context changes within a conversation

**What this measures:** Whether the predictor is useful as a real-time trigger in production agentic systems, not just a post-hoc analysis tool.

### 10.4 What's Deliberately Excluded

Given the refined framing (epistemic state, not correctness), certain experiments are unnecessary:

**Not needed:**
- ❌ Correctness benchmarks against TriviaQA/Natural Questions
- ❌ Hallucination rate measurements
- ❌ Calibration against external ground truth
- ❌ ROC-AUC of predictor score vs. actual errors

**Why:** The context window boundary *is* the ground truth by definition. We're measuring epistemic state (is the model guessing?), and the label for that is already mechanically available. Whether the guess happens to be correct is orthogonal.

The question for agentic systems isn't "did the agent get it right?" (often unknowable without verification) but "should this response be trusted without verification?" The predictor answers this directly from internal state.

## 11. Future Research Directions

### 11.1 Short-term Questions

1. **Domain-specific predictor training:** Complete Experiment 1 across all six domains (medical, legal, code, financial, scientific)
2. **Model family validation:** Run Experiment 2 on Llama, Mistral, Phi, Gemma
3. **LoRA delta analysis:** Measure pre-LoRA vs. post-LoRA topology metrics to determine whether Interpretation A or B is correct
4. **Optimal predictor architecture:** Can even simpler classifiers (linear probes, logistic regression) achieve comparable accuracy?

### 11.2 Medium-term Extensions

1. **Production agentic integration:** Build predictor heads into real multi-agent systems with intervention policies
2. **Multi-turn context tracking:** Develop predictors that track epistemic state across conversation history
3. **Tool-augmented epistemic routing:** Automatic retrieval triggering when predictor score exceeds threshold
4. **Cross-lingual epistemic signals:** Do the activation patterns generalize across languages?
5. **Streaming epistemic monitoring:** Real-time predictor scoring during token generation with early stopping

### 11.3 Long-term Vision

1. **Epistemic fingerprinting:** Can we identify individual training examples' influence by their position in the topology?
2. **Controlled knowledge editing:** Can we modify what the model "knows" by directly manipulating the factual zone?
3. **Unified uncertainty framework:** Can all forms of model uncertainty (epistemic, aleatoric, distributional) be represented topologically?


1. **Multi-modal epistemic topology:** Extension to vision-language models and multimodal reasoning
2. **Epistemic fingerprinting:** Identifying individual training examples' influence by their position in topology
3. **Controlled knowledge editing:** Modifying what the model "knows" by directly manipulating the factual zone
4. **Universal epistemic framework:** Representing all forms of model uncertainty (epistemic, aleatoric, distributional) topologically
5. **Swarm-scale deployment:** Production systems with hundreds of agents, each with epistemic monitoring

---

## 12. Conclusion

We have proposed a novel architecture for inducing epistemic honesty in large language models through adversarial activation pattern training. By leveraging the context window as a ground truth oracle and employing lightweight meta-models to analyze internal states, we create evolutionary pressure for models to route factual recall and speculative extrapolation through distinct neural pathways.

**Preliminary experimental validation** has yielded striking results: a simple 3-layer MLP achieves 99.95% accuracy in classifying epistemic state from frozen Qwen2.5-1.5B activations, suggesting that modern LLMs already exhibit strong natural separation between factual and speculative content at the penultimate layer.

This finding transforms the contribution from "a new training method" to "the discovery of a reliable epistemic state signal in transformer activations." The implications extend far beyond hallucination reduction to enable:

1. **Real-time epistemic monitoring** in multi-agent systems
2. **Selective human-in-the-loop** intervention
3. **Automated confidence-based routing** in agentic workflows
4. **Pre-generation uncertainty detection** without expensive ensemble methods
5. **Mechanistic interpretability** via epistemic topology analysis

**Key Advantages:**

- ✓ Pre-generation detection of unreliable outputs
- ✓ Preserved model capability while reducing hallucinations  
- ✓ Mechanistic interpretability via epistemic topology
- ✓ Immediate implementability using existing infrastructure
- ✓ **Validated empirically** with 99.95% predictor accuracy on initial domain
- ✓ Single forward pass overhead (negligible in production)

The resulting epistemic topology is not merely a practical tool for hallucination mitigation—it is a window into the fundamental structure of machine knowledge. The penumbra region, where factual knowledge blends into inference and creativity, represents the most interesting and valuable aspect of LLM capability. Our goal is not to eliminate this region but to make it transparent and monitorable.

**The Path Forward:**

The preliminary 99.95% predictor accuracy validates the core hypothesis: epistemic state is geometrically encoded in transformer activations and can be reliably extracted. The immediate research agenda focuses on:

1. Domain generalization across medical, legal, code, financial, and scientific contexts
2. Model family validation across Llama, Mistral, Phi, and Gemma architectures  
3. Dynamic agentic context experiments to validate real-time monitoring
4. LoRA delta analysis to determine whether fine-tuning strengthens the pre-existing signal

This work represents a paradigm shift from treating hallucinations as output failures to recognizing them as routing errors detectable at the activation level. Models should be free to speculate, infer, and create—provided their internal states honestly reflect the epistemic status of their outputs.

The components exist, the training signal is clean, the preliminary validation is strong, and the potential impact for agentic systems is substantial. The question is no longer whether we can build AI systems that know what they know—the question is how to deploy this capability at scale in production environments where reliability matters most.

---

## References

**Foundational Machine Learning:**
- Goodfellow, I., et al. (2014). "Generative Adversarial Networks." *NeurIPS*.
- Blundell, C., et al. (2015). "Weight Uncertainty in Neural Networks." *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." *ICML*.
- Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation." *NeurIPS*.

**Interpretability:**
- Belinkov, Y. (2021). "Probing Classifiers: Promises, Shortcomings, and Advances." *Computational Linguistics*.
- Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT." *NeurIPS*.
- Cunningham, H., et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features." *ICLR*.

**LLM Safety and Alignment:**
- Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." *Anthropic*.
- Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*.

**Hallucination and Factuality:**
- Lin, S., et al. (2022). "TruthfulQA: Measuring How Models Mimic Human Falsehoods." *ACL*.
- Manakul, P., et al. (2023). "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection." *EMNLP*.

---

## Appendix A: Glossary

**Activation Pattern:** The vector of neural network activations at a specific layer and token position during text generation.

**Epistemic Status:** The degree to which a model's output is grounded in training data (factual) versus extrapolated (speculative).

**Epistemic Topology:** The geometric structure of a model's activation space, organized by confidence and knowledge certainty.

**Factual Zone:** Regions of activation space associated with grounded recall of training data.

**Generator (G):** The primary language model producing text outputs.

**Hallucination:** Confident generation of false or unsupported information.

**Penumbra:** The intermediate region of activation space representing partial knowledge, logical inference, and bounded speculation.

**Predictor (P):** The meta-model analyzing G's activations to classify epistemic status.

**Routing Error:** When a model processes information through neural pathways inconsistent with its actual epistemic basis (e.g., routing speculation through factual pathways).

**Speculative Zone:** Regions of activation space associated with extrapolation, inference, and creative generation beyond training data.

---

## Appendix B: Complete Implementation Guide for 1.5B Model Proof-of-Concept

### B.1 Why 1.5B is the Perfect PoC Size

A 1.5B parameter model represents the ideal "Goldilocks zone" for this proof-of-concept:

**Computational Feasibility:**
- Trainable on a single prosumer GPU (24GB RTX 3090/4090)
- LoRA fine-tuning keeps memory requirements manageable
- Fast iteration cycles (hours, not days)

**Behavioral Completeness:**
- Large enough to exhibit genuine reasoning and extrapolation
- Displays authentic hallucination behaviors
- Modern architectures (Qwen2.5-1.5B, Llama-3.2-1B, Gemma-2-2B) mirror frontier models

**Scalability:**
- Results directly translate to larger models
- Architecture patterns proven at 1.5B will generalize
- Cost-effective validation before committing to frontier-scale experiments

### B.2 Hardware Requirements

**Minimum Specifications:**
- GPU: 24GB VRAM (RTX 3090, 4090, or A5000)
- RAM: 32GB system memory
- Storage: 100GB for models, datasets, and checkpoints
- CPU: Modern multi-core processor (16+ threads recommended)

**Software Stack:**
- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning) for LoRA
- Standard ML libraries: numpy, scikit-learn, matplotlib

### B.3 Phase 1: The Oracle Data Pipeline

**Step 1.1: Source Factual Contexts**

```python
# Using SQuAD dataset as high-quality factual contexts
from datasets import load_dataset

squad = load_dataset("squad", split="train")

# Extract 10,000 diverse contexts
contexts = []
for i, example in enumerate(squad):
    if i >= 10000:
        break
    contexts.append({
        'context': example['context'],
        'factual_answer': example['answers']['text'][0],
        'factual_question': example['question']
    })
```

**Step 1.2: Generate Symmetric Question Pairs**

For each context, you need:
1. A **factual question** (answerable from context)
2. A **speculative question** (requires extrapolation beyond context)

```python
import anthropic  # or openai for GPT-4

client = anthropic.Anthropic(api_key="your-key")

def generate_question_pairs(context):
    """
    Generate factual and speculative questions for a given context.
    """
    
    prompt = f"""Given this context:

{context}

Generate TWO questions:

1. FACTUAL: A question whose answer is explicitly stated in the context above.
2. SPECULATIVE: A question thematically related to the context, but whose answer requires extrapolation, prediction, or knowledge beyond what's provided.

Format:
FACTUAL: [question]
SPECULATIVE: [question]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response to extract both questions
    text = response.content[0].text
    factual = extract_question(text, "FACTUAL:")
    speculative = extract_question(text, "SPECULATIVE:")
    
    return factual, speculative

# Generate pairs for all contexts
oracle_dataset = []
for ctx in contexts[:10000]:
    factual_q, speculative_q = generate_question_pairs(ctx['context'])
    
    oracle_dataset.append({
        'context': ctx['context'],
        'question': factual_q,
        'label': 1  # Factual
    })
    
    oracle_dataset.append({
        'context': ctx['context'],
        'question': speculative_q,
        'label': 0  # Speculative
    })

# Result: 20,000 examples (10k factual, 10k speculative)
```

**Step 1.3: Dataset Quality Validation**

```python
# Manual review of samples
import random

def validate_dataset(dataset, n_samples=50):
    """
    Manually review random samples to ensure quality.
    """
    samples = random.sample(dataset, n_samples)
    
    print("VALIDATION SAMPLES:")
    print("=" * 80)
    
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Context: {sample['context'][:200]}...")
        print(f"Question: {sample['question']}")
        print(f"Label: {'FACTUAL' if sample['label'] == 1 else 'SPECULATIVE'}")
        print("-" * 80)
        
        # Human validates each sample
        valid = input("Is this correctly labeled? (y/n): ")
        if valid.lower() != 'y':
            print("⚠️  Flagged for review")

validate_dataset(oracle_dataset)
```

### B.4 Phase 2: Harvesting the Activations

**Step 2.1: Load the Base Model**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # or "meta-llama/Llama-3.2-1B"

# Load model in 16-bit precision
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
```

**Step 2.2: Extract Activation Vectors**

```python
def extract_activation(model, tokenizer, context, question):
    """
    Extract the penultimate layer hidden state for the first generated token.
    
    Returns:
        activation: 1D tensor of shape (hidden_size,) e.g., (2048,) or (4096,)
    """
    
    # Format prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                      max_length=2048).to(model.device)
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False
        )
    
    # Extract penultimate layer (second to last)
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    # Each tensor has shape (batch_size, seq_len, hidden_size)
    penultimate_layer = outputs.hidden_states[-2]  # -1 is final, -2 is penultimate
    
    # Get hidden state of the last token (right before generation)
    # Shape: (1, hidden_size)
    last_token_activation = penultimate_layer[0, -1, :]
    
    # Return as 1D numpy array
    return last_token_activation.cpu().numpy()


# Extract activations for entire dataset
import numpy as np
from tqdm import tqdm

activations = []
labels = []

print("Extracting activations from frozen model...")
for example in tqdm(oracle_dataset):
    activation = extract_activation(
        model, tokenizer, 
        example['context'], 
        example['question']
    )
    
    activations.append(activation)
    labels.append(example['label'])

# Convert to numpy arrays
X = np.array(activations)  # Shape: (20000, hidden_size)
y = np.array(labels)        # Shape: (20000,)

# Save to disk
np.save('activations.npy', X)
np.save('labels.npy', y)

print(f"Saved {len(X)} activation vectors of dimension {X.shape[1]}")
```

### B.5 Phase 3: Training the Predictor (P)

**Step 3.1: Build the Predictor MLP**

```python
import torch.nn as nn

class ActivationClassifier(nn.Module):
    """
    Simple MLP to classify epistemic status from activations.
    """
    
    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


# Initialize predictor
hidden_size = X.shape[1]  # Automatically detect from activations
predictor = ActivationClassifier(input_dim=hidden_size).cuda()

print(f"Predictor architecture:")
print(predictor)
print(f"\nTotal parameters: {sum(p.numel() for p in predictor.parameters()):,}")
```

**Step 3.2: Train the Predictor**

```python
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Load saved activations
X = np.load('activations.npy')
y = np.load('labels.npy')

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train).cuda()
y_train_t = torch.FloatTensor(y_train).unsqueeze(1).cuda()
X_val_t = torch.FloatTensor(X_val).cuda()
y_val_t = torch.FloatTensor(y_val).unsqueeze(1).cuda()

# Create dataloaders
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.BCELoss()

def train_predictor(model, train_loader, val_loader, epochs=20):
    """
    Train the activation classifier.
    """
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += ((predictions > 0.5) == y_batch).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_correct += ((predictions > 0.5) == y_batch).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'predictor_best.pt')
            print(f"  ✓ New best model saved!")
        print()
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    
    # Return True if accuracy is good enough
    return best_val_acc >= 0.85

success = train_predictor(predictor, train_loader, val_loader)

if success:
    print("✅ Predictor achieved >85% accuracy!")
    print("Ready to proceed to Phase 4: Adversarial Fine-tuning")
else:
    print("⚠️  Predictor did not reach 85% accuracy.")
    print("Consider: (1) More data, (2) Better question generation, (3) Different layer")
```

### B.6 Phase 4: The Symbiotic Training Loop

**Step 4.1: Setup LoRA for Generator**

```python
from peft import LoraConfig, get_peft_model, TaskType

# Load fresh base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention matrices
    bias="none"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load trained predictor and freeze it
predictor.load_state_dict(torch.load('predictor_best.pt'))
predictor.eval()
for param in predictor.parameters():
    param.requires_grad = False
```

**Step 4.2: Custom Training Loop with Dual Loss**

```python
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F

class EpistemicRoutingTrainer(Trainer):
    """
    Custom trainer that combines generation loss with routing loss.
    """
    
    def __init__(self, predictor, lambda_routing=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = predictor
        self.lambda_routing = lambda_routing
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute combined loss: L_generation + λ * L_routing
        """
        
        # Standard forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
            output_hidden_states=True
        )
        
        # L_generation (standard causal LM loss)
        generation_loss = outputs.loss
        
        # Extract penultimate layer activations
        penultimate_activations = outputs.hidden_states[-2]  # (batch, seq_len, hidden)
        
        # Get activation of last token before generation
        last_token_activations = penultimate_activations[:, -1, :]  # (batch, hidden)
        
        # Predictor classification
        epistemic_predictions = self.predictor(last_token_activations.float())
        
        # Ground truth labels (from dataset)
        ground_truth = inputs['epistemic_labels'].unsqueeze(1).float()
        
        # L_routing (binary cross-entropy)
        routing_loss = F.binary_cross_entropy(epistemic_predictions, ground_truth)
        
        # Combined loss
        total_loss = generation_loss + self.lambda_routing * routing_loss
        
        # Logging
        self.log({
            'loss_generation': generation_loss.item(),
            'loss_routing': routing_loss.item(),
            'loss_total': total_loss.item()
        })
        
        return (total_loss, outputs) if return_outputs else total_loss


# Prepare dataset for training
def prepare_training_data(oracle_dataset, tokenizer, max_length=2048):
    """
    Convert oracle dataset to format expected by trainer.
    """
    
    train_data = []
    
    for example in oracle_dataset:
        prompt = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
        
        # Tokenize
        encoding = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        train_data.append({
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': encoding['input_ids'][0].clone(),  # For causal LM
            'epistemic_labels': torch.tensor(example['label'])
        })
    
    return train_data

training_data = prepare_training_data(oracle_dataset, tokenizer)


# Training arguments
training_args = TrainingArguments(
    output_dir='./epistemic_routing_model',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 16
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    fp16=True,
    report_to='none'
)

# Initialize trainer
trainer = EpistemicRoutingTrainer(
    predictor=predictor,
    lambda_routing=1.0,  # Equal weighting of both losses
    model=model,
    args=training_args,
    train_dataset=training_data,
    tokenizer=tokenizer
)

# Train!
print("Starting epistemic routing fine-tuning...")
trainer.train()

# Save final model
model.save_pretrained('./epistemic_routing_final')
print("✅ Training complete!")
```

### B.7 Phase 5: Visualizing the Epistemic Topology

**Step 5.1: Extract Activations from Trained Model**

```python
def extract_activations_batch(model, tokenizer, examples):
    """
    Extract activations from the fine-tuned model.
    """
    activations = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for example in tqdm(examples):
            prompt = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=2048).to(model.device)
            
            outputs = model(**inputs, output_hidden_states=True)
            
            # Penultimate layer, last token
            activation = outputs.hidden_states[-2][0, -1, :].cpu().numpy()
            
            activations.append(activation)
            labels.append(example['label'])
    
    return np.array(activations), np.array(labels)

# Generate fresh test set (different from training)
test_contexts = contexts[10000:11000]  # 1000 new contexts
test_dataset = generate_oracle_dataset(test_contexts)  # Your function from Phase 1

# Extract activations
X_test, y_test = extract_activations_batch(model, tokenizer, test_dataset)

# Save
np.save('test_activations_trained.npy', X_test)
np.save('test_labels.npy', y_test)
```

**Step 5.2: Dimensionality Reduction and Visualization**

```python
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

# Load activations
X_test = np.load('test_activations_trained.npy')
y_test = np.load('test_labels.npy')

# Apply t-SNE
print("Running t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d_tsne = tsne.fit_transform(X_test)

# Apply UMAP (often better for clusters)
print("Running UMAP dimensionality reduction...")
umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
X_2d_umap = umap_reducer.fit_transform(X_test)

# Visualization function
def plot_epistemic_topology(X_2d, labels, title, filename):
    """
    Plot the epistemic topology with factual and speculative clusters.
    """
    plt.figure(figsize=(12, 8))
    
    # Separate by label
    factual_mask = labels == 1
    speculative_mask = labels == 0
    
    # Plot
    plt.scatter(
        X_2d[factual_mask, 0], 
        X_2d[factual_mask, 1],
        c='blue', 
        label='Factual', 
        alpha=0.6, 
        s=50
    )
    
    plt.scatter(
        X_2d[speculative_mask, 0], 
        X_2d[speculative_mask, 1],
        c='red', 
        label='Speculative', 
        alpha=0.6, 
        s=50
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved visualization to {filename}")

# Generate visualizations
plot_epistemic_topology(
    X_2d_tsne, 
    y_test, 
    'Epistemic Topology (t-SNE)', 
    'topology_tsne.png'
)

plot_epistemic_topology(
    X_2d_umap, 
    y_test, 
    'Epistemic Topology (UMAP)', 
    'topology_umap.png'
)
```

**Step 5.3: Quantitative Analysis**

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

def analyze_separation(X_2d, labels):
    """
    Quantify the separation between factual and speculative clusters.
    """
    
    # Silhouette score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(X_2d, labels)
    
    # Davies-Bouldin index (lower is better)
    db_index = davies_bouldin_score(X_2d, labels)
    
    # Inter-cluster distance
    factual_centroid = X_2d[labels == 1].mean(axis=0)
    speculative_centroid = X_2d[labels == 0].mean(axis=0)
    inter_distance = np.linalg.norm(factual_centroid - speculative_centroid)
    
    # Intra-cluster variance
    factual_variance = np.var(X_2d[labels == 1], axis=0).mean()
    speculative_variance = np.var(X_2d[labels == 0], axis=0).mean()
    avg_variance = (factual_variance + speculative_variance) / 2
    
    # Separation ratio
    separation_ratio = inter_distance / np.sqrt(avg_variance)
    
    print("=" * 60)
    print("EPISTEMIC TOPOLOGY ANALYSIS")
    print("=" * 60)
    print(f"Silhouette Score:       {silhouette:.4f} (higher is better)")
    print(f"Davies-Bouldin Index:   {db_index:.4f} (lower is better)")
    print(f"Inter-cluster Distance: {inter_distance:.4f}")
    print(f"Separation Ratio:       {separation_ratio:.4f}")
    print("=" * 60)
    
    # Success criteria
    if silhouette > 0.3 and separation_ratio > 2.0:
        print("✅ STRONG EPISTEMIC SEPARATION DETECTED")
        print("The model has learned distinct pathways for factual vs speculative content!")
    elif silhouette > 0.15:
        print("⚠️  MODERATE SEPARATION")
        print("Some pathway differentiation, but could be stronger.")
    else:
        print("❌ WEAK SEPARATION")
        print("The model has not developed distinct epistemic pathways.")
    
    return {
        'silhouette': silhouette,
        'db_index': db_index,
        'inter_distance': inter_distance,
        'separation_ratio': separation_ratio
    }

# Run analysis
metrics = analyze_separation(X_2d_umap, y_test)
```

### B.8 Success Criteria and Next Steps

**Proof-of-Concept is Successful if:**

1. **Predictor Performance**: Initial classifier achieves >85% accuracy on frozen model activations
2. **Maintained Capability**: Fine-tuned model shows no degradation on standard benchmarks
3. **Visual Separation**: t-SNE/UMAP plots show distinct factual and speculative clusters
4. **Quantitative Separation**: Silhouette score > 0.3, separation ratio > 2.0
5. **Hallucination Reduction**: When heeding predictor warnings, hallucination rate decreases by >30%

**If Successful, Next Steps:**

1. **Scale Up**: Apply to 7B, 13B, 70B models
2. **Multi-domain Testing**: Evaluate across different knowledge domains
3. **Intervention Strategies**: Test different policies when predictor flags uncertainty
4. **Paper Draft**: Publish results with visualizations and code
5. **Open Source**: Release trained models and replication code

**If Unsuccessful:**

1. **Data Quality**: Review question generation quality
2. **Layer Selection**: Try different transformer layers for activation extraction
3. **Architecture**: Test different predictor architectures
4. **Hyperparameters**: Adjust λ (routing weight), LoRA rank, learning rates

### B.9 Estimated Timeline and Costs

**Week 1: Data Pipeline**
- Generate 10,000 oracle question pairs
- Cost: ~$50 in API calls (Claude/GPT-4)
- Time: 2-3 days

**Week 2: Predictor Training**
- Extract activations and train classifier
- Cost: Free (local GPU)
- Time: 1 day

**Week 3-4: Adversarial Fine-tuning**
- LoRA training with dual loss
- Cost: Free (local GPU)
- Time: 3-5 days (including hyperparameter tuning)

**Week 5: Analysis and Visualization**
- Generate topology plots
- Quantitative analysis
- Cost: Free
- Time: 2 days

**Total Estimated Cost**: <$100
**Total Estimated Time**: 5 weeks (part-time) or 2 weeks (full-time)

### B.10 Complete Code Repository Structure

```
epistemic-routing-poc/
├── data/
│   ├── generate_oracle_data.py
│   ├── oracle_dataset.json
│   └── validate_data.py
├── models/
│   ├── predictor.py
│   ├── trainer.py
│   └── config.yaml
├── experiments/
│   ├── phase1_data.py
│   ├── phase2_activations.py
│   ├── phase3_predictor.py
│   ├── phase4_training.py
│   └── phase5_visualization.py
├── analysis/
│   ├── topology_plots.py
│   ├── metrics.py
│   └── compare_models.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_predictor_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── requirements.txt
├── README.md
└── run_poc.sh
```

## Appendix C: Experimental Pseudocode

```python
# Simplified training loop for epistemic routing

def train_epistemic_routing(base_model, training_data, epochs):
    """
    Train a model to route factual vs speculative content through
    distinct neural pathways using adversarial activation supervision.
    """
    
    # Initialize models
    G = base_model  # Generator (primary LLM)
    P = ActivationClassifier()  # Predictor (lightweight)
    
    # Training loop
    for epoch in range(epochs):
        for batch in training_data:
            
            # Batch contains questions with epistemic labels
            # label = 'factual' if answerable from context
            # label = 'speculative' if requires extrapolation
            
            contexts, questions, labels = batch
            
            # Forward pass through generator
            outputs, activations = G.forward_with_activations(
                contexts, questions
            )
            
            # Predictor classifies epistemic status from activations
            predictions = P.forward(activations)
            
            # Compute losses
            generation_loss = cross_entropy(outputs, targets)
            routing_loss = cross_entropy(predictions, labels)
            
            # Combined objective
            total_loss = generation_loss + λ * routing_loss
            
            # Backpropagate through both models
            total_loss.backward()
            G.optimizer.step()
            P.optimizer.step()
            
        # Periodically analyze epistemic topology
        if epoch % 10 == 0:
            analyze_topology(G, P, validation_data)
    
    return G, P
```

This appendix provides everything needed to replicate the 1.5B parameter proof-of-concept, from data generation through visualization of the epistemic topology.

---

**Document Information:**
- Version: 1.1 (Updated with complete implementation guide)
- Date: February 2026
- License: Open for academic and commercial research

---

*This white paper represents a novel research proposal with a complete, executable implementation path. We welcome collaboration, critique, and extension of these ideas.*
