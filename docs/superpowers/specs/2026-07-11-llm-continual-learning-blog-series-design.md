# LLM Continual Learning Blog Series Design

## Goal

Create two connected Chinese long-form blog posts for readers who understand Transformer basics but are new to LoRA and continual learning. The posts should serve both as an introduction and as a research-oriented path toward the author's paper on orthogonal LoRA instability.

## Shared Writing Principles

- Explain each idea in the order: problem, intuition, mathematics, training procedure, experiments, limitations.
- Define symbols before using them and connect equations to plain-language interpretation.
- Distinguish foundational continual-learning methods from methods designed specifically for language models.
- For every deeply discussed paper, report its datasets, models, baselines, metrics, main results, and limitations.
- Use comparison tables to prevent the article from becoming a sequence of isolated summaries.
- Keep English method names and technical terms where they improve precision, with Chinese explanations around them.

## Post 1: Foundations and Classic Directions

Working title: `语言模型如何持续学习：从灾难性遗忘到重放、正交子空间与 LoRA`

### Scope

1. Formalize sequential learning, catastrophic forgetting, and the stability-plasticity dilemma.
2. Explain task-, domain-, and class-incremental settings and how language-model continual learning differs from conventional classification.
3. Define Average Accuracy, Forgetting Measure, BWT, FWT, parameter growth, replay requirements, and task-ID assumptions.
4. Build a taxonomy covering replay, regularization, gradient/subspace constraints, parameter isolation/expansion, and PEFT/LoRA.
5. Deeply discuss four representative language-model papers:
   - LAMOL
   - LFPT5
   - Progressive Prompts
   - O-LoRA
6. End with the unresolved assumptions behind orthogonal LoRA and transition into Post 2.

Foundational methods such as EWC, LwF, GEM, and OGD will receive mathematical explanations but not full paper-level experimental profiles, because they are not all language-model-specific papers.

## Post 2: Recent Orthogonal and Subspace Methods

Working title: `正交 LoRA 真能防止遗忘吗：从子空间复用到双线性优化失稳`

### Scope

Use five papers to form a single research progression:

1. C-LoRA: single-LoRA routing, subspace reuse, and orthogonality.
2. OA-Adapter: adaptive parameter budgets across tasks and layers.
3. OLieRA: multiplicative low-rank updates on Lie groups and parameter geometry.
4. ELLA: selective subspace de-correlation and the plasticity cost of strict orthogonality.
5. The author's SFOR paper: BOD, WRP, Scaling Buffer Effect, strict versus mature SFOR, and safe basis expansion.

For each paper, include method equations, optimization intuition, datasets, models, baselines, metrics, major results, and critical limitations. Mention Sculpting Subspaces, DOC, and SLICE as related comparisons without turning them into full paper profiles.

### Narrative

The article moves through five questions:

1. Can one shared LoRA reuse knowledge without one adapter per task?
2. How much orthogonal capacity should each task and layer receive?
3. Does additive adaptation preserve the geometry of model parameters?
4. Does strict orthogonality destroy useful transfer and exhaust plasticity?
5. Even if routing updates are orthogonal, can LoRA's unconstrained bilinear factor and AdamW still bypass the protection?

The final synthesis compares memory, parameter growth, replay, task-ID requirements, geometric constraint, plasticity, and optimizer sensitivity.

## Deliverables

- Two publishable Markdown posts under `src/content/posts/`.
- Correct Astro frontmatter matching the existing content schema.
- Inline LaTeX equations and compact comparison tables.
- References with direct arXiv or official publication links.
- A successful production build after both posts are added.
