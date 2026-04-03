# Project Name
LexiGen: A Synthetic Language Translation Benchmark Generator

# Your Team
Michał Gromadzki

# Problem Statement
A major open problem in evaluating large language models is distinguishing true reasoning and compositional generalization from memorization and pattern matching. 

The ARC-AGI-2 Benchmark (Chollet et al., 2025) partially addresses this problem by testing reasoning and abstraction through program-like tasks. However, ARC-AGI tasks are based on 2D colored grids, which are visually and spatially structured. While effective for general intelligence evaluation, this representation is not well aligned with language models, which operate primarily on sequences of tokens rather than spatial grids. As a result, ARC-style tasks may underestimate reasoning ability in language models due to the mismatch in representation rather than actual reasoning limitations.

LexiGen is designed to address this gap by creating an ARC-like reasoning benchmark in a language domain instead of a visual grid domain. Instead of transforming 2D grids, models must infer the rules of an invented language from a small set of examples and apply those rules to translate new sentences.

This setup tests whether models can:
- Infer grammar rules from examples
- Apply morphological transformations (tense, plural)
- Handle compositional sentence structure
- Generalize rules to new combinations of known words
- Translate multi-clause sentences

Unlike traditional translation datasets, the language in LexiGen is synthetic and rule-based, meaning there is no memorization advantage. The only way to succeed is to infer the underlying grammar rules from the examples.

# Task & Benchmark Construction
Each task in LexiGen is a few-shot translation problem consisting of:
- **5 training examples** (English → synthetic language)
- **1 test example** requiring translation from English to synthetic language

Tasks are generated programmatically using a controlled vocabulary and grammar rules. The artificial language includes:
- Prefix-based verb conjugation (tense)
- Vowel-shift pluralization
- Suffix-based articles
- Fixed word order (Subject-Verb-Object)
- Optional negation and question markers
- Multi-clause composition via conjunctions

To ensure tasks are valid and solvable, we enforce three constraints:

1. Vocabulary Coverage - All base words in the test sentence must appear in the training examples.
2. Feature Coverage - 
All grammatical features in the test sentence must be demonstrated in the examples:
    - Tense (past, present, future)
    - Negation
    - Copula constructions
    - Adjectives
    - Conjunctions (multi-clause structure)

3. Morphological Rule Coverage - Plural formation is implemented via deterministic vowel shifts. The examples must include the same vowel transformation patterns required for the test.


Tasks that fail any constraint are discarded and regenerated, ensuring:
- No ambiguity in correct answers
- Strong alignment between training signal and test requirements

# Dataset
The dataset consists of **100 automatically generated tasks**, stored as a JSON file.

Each task contains:
- `train`: list of 5 (English, synthetic) sentence pairs
- `test`: a single (English, synthetic) pair

Example format:
```json
{
  "train": [
    ["The dog eats the apple", "la-morin den sog en telar"],
    ...
  ],
  "test": [
    ["The dogs will eat the apple", "ka-morin du sogu en telar"]
  ]
}
```

# Technical Details
LexiGen combines symbolic generation, NLP parsing, and deterministic grammar rules to produce synthetic translation tasks. The pipeline is designed to create **high-quality, verifiable tasks** that test compositional generalization.

## 1. Vocabulary and Grammar Rule Generation
- LexiGen samples a **task-local vocabulary** with fixed defaults:
  - 12 subjects
  - 12 objects
  - 10 verbs
  - 10 adjectives
  These are sampled from global lists in `constants.py`.
- English base words are mapped to synthetic tokens through a persistent dictionary:
  - A word gets assigned once and reused everywhere in the same task.
  - New tokens are generated with lengths between 3 and 8 characters.
  - Token shape alternates consonants/vowels with a stochastic toggle (80% chance to switch class each step), producing pronounceable pseudo-words.

Grammar transformations are deterministic:
- **Verb tense**:
  - present -> `la-` + base verb token
  - past -> `na-` + base verb token
  - future -> `ka-` + base verb token
- **Plural nouns**: first matching vowel is shifted by a cyclic mapping
  - `a->e`, `e->i`, `i->o`, `o->u`, `u->a`
  - if no vowel is found, `e` is appended
- **Articles**: noun token gets suffix
  - singular -> `en`
  - plural -> `u`
- **Negation**: sentence-level prefix `no `
- **Question marking**: sentence-level suffix ` va`
- **Conjunction lexicon**:
  - `and -> sa`, `because -> ko`, `but -> ba`, `or -> ra`

Sentence generation for dataset creation uses:
- simple English clauses sampled from the task-local vocabulary
- sentences with exactly one additional clause, joined by either `and` or `because`

## 2. Sentence Parsing and Feature Extraction
Parsing and translation analysis are implemented with `spaCy`.

Per-clause extraction logic:
- **Subject**: first token with dependency `nsubj` or `nsubjpass`; fallback is first pronoun/noun/proper noun.
- **Object**: first token with dependency `dobj` or `pobj`.
- **Verb and tense**: first `VERB`/`AUX` token is used.
  - `VBD`/`VBN` -> past
  - `VB`/`VBP`/`VBZ` -> present
  - `MD` -> future
- **Negation**: any token with dependency `neg` sets negation flag.
- **Adjectives**: `ADJ` tokens headed by nouns/proper nouns are attached to the corresponding noun phrase.
- **Copula pattern**: adjective complements (`acomp`) are detected; when verb lemma is `be`, output is rendered as subject phrase + adjective token (instead of standard verb-subject-object form).

Token normalization and morphology handling:
- Nouns/proper nouns use lemma-based base forms.
- Plural detection uses POS tags `NNS` and `NNPS`.
- Noun phrase construction applies base-token lookup, plural transformation, article suffixing, and attached adjectives.

Multi-clause handling:
- Clauses are split on conjunction POS (`CCONJ`/`SCONJ`) and punctuation tokens are removed.
- Conjunction words are mapped to ISL equivalents through the conjunction lexicon.
- Each clause is translated independently, then clauses are joined with mapped conjunction tokens.

## 3. Sentence Translation
Translation is done clause by clause and then merged into a final sentence.

Core process:
- Parse each clause to identify subject, object, verb, tense, adjectives, and negation.
- Build noun phrases using base-token mapping, plural transformation, and article suffixes.
- Conjugate verbs by tense and assemble output in Verb-Subject-(Object) order.
- Apply sentence markers: `no ` for negation and ` va` for questions.
- Handle copula clauses (`be` + adjective) with a simplified subject-adjective form.
- Join translated clauses with mapped conjunction tokens.

Plural metadata is tracked during translation and reused later for rule-coverage validation.

## 4. Task Validation
Each task is checked to ensure it is solvable:
- Vocabulary coverage: all test words appear in training
- Feature coverage: all grammatical features appear in training
- Plural rule coverage: vowel-shift patterns demonstrated in training

## 5. Dataset Loading and Structure
- Tasks are stored in a JSON file (`isl_dataset.json`) with 100 generated tasks.
- Each task contains:
  - `train`: list of 5 English-ISL sentence pairs
  - `test`: a single English-ISL sentence pair

Tasks are converted into a tabular format with:
- `task_index`
- `full_prompt` (few-shot translation prompt)
- `user_request` (test sentence)
- `context_document` (concatenated training examples)
- `expected_translation` (ground truth ISL translation)

## 6. Prompt Construction
Few-shot prompts include all training examples plus test sentence:

```text
You translate English sentences into an invented sign language.
Infer the translation rules only from the examples.
Return only the translated sentence as lowercase tokens separated by spaces.
Do not add explanations, punctuation, quotes, or labels.

Examples:
[source sentence 1] -> [target sentence 1]
[source sentence 2] -> [target sentence 2]
...

Sentence: [test sentence]
Translation:
```

## 7. Pipeline Summary
1. Sample vocabulary -> generate English sentences -> parse with spaCy
2. Translate sentences -> apply deterministic grammar rules -> create ISL sentences
3. Validate tasks -> ensure coverage and rule applicability
4. Build few-shot prompts -> feed to LLM -> normalize output -> compare with expected translation

# Results, Insights, and Conclusions

The benchmark remains difficult for most models: performance clusters near zero, with only a small number of systems reaching modest gains.

## Results
- Stronger reasoning-focused models perform slightly better, but the improvement is limited.
- The task still exposes a wide gap between frontier models and reliable compositional translation.

## Insights
- The benchmark isolates rule induction and compositional generalization more than memorization.
- Performance remains low overall, which suggests the task is genuinely difficult for current models.
- The synthetic grammar provides a controlled way to compare reasoning ability across model families.

## Future Work
- The benchmark could be augmented with intentionally unsolvable tasks to better probe model robustness.
- It could also include metacognition-oriented tasks, where models must recognize when a translation is likely impossible or underspecified.
- Adding many irrelevant or distracting examples could help measure attention, filtering, and resistance to noise.

# Organizational Affiliations
Warsaw University of Technology

# References & Citations
Chollet, F., Knoop, M., Kamradt, G., Landers, B., & Pinkard, H. (2025).  ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems. arXiv:2505.11831.
