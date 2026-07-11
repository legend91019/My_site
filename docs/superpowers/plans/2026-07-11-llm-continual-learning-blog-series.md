# LLM Continual Learning Blog Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish two connected Chinese blog posts that introduce language-model continual learning and trace recent orthogonal LoRA research into the author's SFOR paper.

**Architecture:** Each post is a standalone Astro content entry under `src/content/posts/`. Research notes are temporary working material; claims in the published posts must be traceable to the source papers. The first post builds the field taxonomy and classic-paper foundation, while the second uses five papers as a chronological and conceptual argument.

**Tech Stack:** Astro content collections, Markdown, KaTeX-compatible LaTeX, npm production build.

## Global Constraints

- Write for readers who understand Transformer basics but are new to LoRA and continual learning.
- Explain deep paper profiles as problem, intuition, mathematics, training procedure, experiments, results, and limitations.
- Report datasets, models, baselines, metrics, and parameter-growth or replay assumptions for every deep paper profile.
- Use direct arXiv or official publication links and do not invent numerical results that are absent from the source.
- Match the frontmatter schema in `src/content.config.ts`.

---

### Task 1: Build the Classic-Paper Evidence Matrix

**Files:**
- Read: `src/content.config.ts`
- Read: source PDFs or official paper pages for LAMOL, LFPT5, Progressive Prompts, and O-LoRA
- Temporary: local extracted paper text outside `src/content/posts/`

**Interfaces:**
- Consumes: the paper titles fixed in the design spec.
- Produces: verified notes for method equations, datasets, models, baselines, metrics, headline results, and limitations.

- [ ] **Step 1: Acquire the four official papers**

Use arXiv metadata to identify the canonical IDs, then download the PDFs into a temporary research directory.

- [ ] **Step 2: Extract searchable text**

Run `pdftotext -layout` for each PDF and confirm that the abstract, methodology, experimental setup, and conclusion are present.

- [ ] **Step 3: Record a uniform evidence matrix**

For each paper, capture: publication year, continual-learning setting, replay policy, parameter-growth behavior, task-ID assumption, core objective, exact source equation, datasets, base model, baselines, metrics, main numerical result, and one limitation.

- [ ] **Step 4: Cross-check foundational formulas**

Verify EWC, LwF, GEM, OGD, Average Accuracy, Forgetting, BWT, and FWT against their original definitions or a cited survey.

### Task 2: Write the Foundations Post

**Files:**
- Create: `src/content/posts/llm_continual_learning_foundations.md`
- Reference: `src/content/posts/llm_components.md`
- Reference: `src/content/posts/tokenizer.md`

**Interfaces:**
- Consumes: the verified classic-paper evidence matrix from Task 1.
- Produces: a publishable post whose closing questions introduce the assumptions examined in Post 2.

- [ ] **Step 1: Add valid frontmatter**

Use a Chinese title and summary, date `2026-07-11`, tags for `大模型`, `持续学习`, and `灾难性遗忘`, category `学习指南`, and `draft: false`.

- [ ] **Step 2: Establish the formal problem and metrics**

Define task sequences and stability-plasticity, then give equations for Average Accuracy, Forgetting Measure, BWT, and FWT with symbol explanations.

- [ ] **Step 3: Explain the five-method taxonomy**

Cover replay, regularization, gradient/subspace constraints, parameter isolation or expansion, and PEFT/LoRA. Include EWC, LwF, GEM, and OGD equations as foundational examples.

- [ ] **Step 4: Add four complete paper profiles**

Write LAMOL, LFPT5, Progressive Prompts, and O-LoRA using the shared profile template. Each profile must include models, datasets, baselines, experimental conclusion, and limitations.

- [ ] **Step 5: Add comparison and transition sections**

Include a compact route comparison table and end by identifying the bilinear and optimizer assumptions left unresolved by O-LoRA.

- [ ] **Step 6: Validate the first post**

Run `npm run build`. Expected: Astro exits with code 0 and emits the new post route without content-schema errors.

### Task 3: Build the Recent-Paper Evidence Matrix

**Files:**
- Read: official papers for Sculpting Subspaces, OA-Adapter, OLieRA, and ELLA
- Read: `D:/Desktop/Research/On the Instability of Orthogonal LoRA in Continual Learning3.pdf`
- Temporary: local extracted paper text outside `src/content/posts/`

**Interfaces:**
- Consumes: the five-paper sequence fixed in the design spec.
- Produces: verified equations and experiment cards for Post 2.

- [ ] **Step 1: Acquire and extract the four external papers**

Download their official arXiv PDFs and extract layout-preserving text.

- [ ] **Step 2: Extract all five method definitions**

Capture adaptive-SVD constraints for Sculpting Subspaces, dynamic bottleneck budgeting for OA-Adapter, Lie-group multiplicative adaptation for OLieRA, selective subspace de-correlation for ELLA, and BOD/SFOR/WRP/Mature SFOR for the author's paper.

- [ ] **Step 3: Extract experiment cards**

For each paper, record datasets, task sequence, model architecture and size, baselines, metrics, strongest reported result, ablations, and limitations.

- [ ] **Step 4: Verify comparison fairness**

Mark results that cannot be compared directly because they use different models, datasets, task orders, parameter budgets, or replay assumptions.

### Task 4: Write the Recent Orthogonal-LoRA Post

**Files:**
- Create: `src/content/posts/orthogonal_lora_continual_learning.md`
- Reference: `src/content/posts/llm_continual_learning_foundations.md`

**Interfaces:**
- Consumes: the verified five-paper evidence matrix from Task 3.
- Produces: a publishable research-oriented post ending in the author's SFOR contribution.

- [ ] **Step 1: Add valid frontmatter**

Use a Chinese title and summary, date `2026-07-11`, tags for `大模型`, `持续学习`, `LoRA`, and `正交子空间`, category `论文解读`, and `draft: false`.

- [ ] **Step 2: Reintroduce LoRA geometry**

Define `W = W_0 + BA`, explain basis versus routing interpretations, and state why linear orthogonality arguments are not automatically sufficient for a bilinear parameterization.

- [ ] **Step 3: Write the Sculpting Subspaces, OA-Adapter, OLieRA, and ELLA profiles**

For each profile, include its core equation, method intuition, training flow, experiments, baselines, model choice, results, and a transition question leading to the next paper.

- [ ] **Step 4: Write the SFOR profile as the main section**

Explain the Taylor assumptions, BOD compensation path, AdamW tearing, semi-frozen routing, WRP, Scaling Buffer Effect, adaptive subspace budgeting, and safe orthogonal basis expansion. Include the 3B/7B, five-task, TRACE, and long-horizon evidence reported in the paper.

- [ ] **Step 5: Add a cross-paper synthesis**

Compare replay, parameter growth, task-ID dependence, constraint target, geometry, plasticity, optimizer sensitivity, models, datasets, and evaluation metrics. Explicitly distinguish cross-paper evidence from within-paper comparisons.

- [ ] **Step 6: Add references and critical conclusions**

Link all five papers and briefly position C-LoRA, DOC, and SLICE as nearby work. End with open questions rather than claiming the field is solved.

### Task 5: Verify and Clean the Series

**Files:**
- Verify: `src/content/posts/llm_continual_learning_foundations.md`
- Verify: `src/content/posts/orthogonal_lora_continual_learning.md`
- Verify: `docs/superpowers/specs/2026-07-11-llm-continual-learning-blog-series-design.md`

**Interfaces:**
- Consumes: both completed posts.
- Produces: build-verified publishable content with no temporary research files tracked by Git.

- [ ] **Step 1: Scan for incomplete or unsupported prose**

Run `rg -n "TBD|TODO|待补|据说|显著提升" src/content/posts/llm_continual_learning_foundations.md src/content/posts/orthogonal_lora_continual_learning.md` and resolve every placeholder or unsupported comparative claim.

- [ ] **Step 2: Check links, headings, and frontmatter**

Confirm every deep paper has a reference link and every frontmatter field passes the content schema.

- [ ] **Step 3: Run the production build**

Run `npm run build`. Expected: exit code 0, both post routes generated, no Markdown or schema errors.

- [ ] **Step 4: Review the rendered pages**

Start the local server and inspect desktop and mobile layouts for table overflow, equation clipping, malformed characters, and heading hierarchy.

- [ ] **Step 5: Review the final diff**

Run `git diff --check` and `git status --short`. Expected: only the two posts, plan documentation, and intentional assets are changed; temporary PDFs and extracted text remain untracked or are removed.
