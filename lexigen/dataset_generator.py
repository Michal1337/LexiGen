import json
import random
from pathlib import Path
from tqdm import trange

import inflect
import spacy

from constants import (
    ADJECTIVES,
    CONJUNCTIONS,
    CONSONANTS,
    OBJECTS,
    SUBJECTS,
    VERBS,
    VOWELS,
)

random.seed(1337)
p = inflect.engine()
nlp = spacy.load("en_core_web_sm")
dictionary = {}


def generate_wordlike_token(min_len=3, max_len=8):
    length = random.randint(min_len, max_len)
    token = []
    use_consonant = random.choice([True, False])
    for _ in range(length):
        if use_consonant:
            token.append(random.choice(CONSONANTS))
        else:
            token.append(random.choice(VOWELS))
        if random.random() < 0.8:
            use_consonant = not use_consonant
    return "".join(token)


def get_base_word(word):
    word = word.lower()
    if word not in dictionary:
        dictionary[word] = generate_wordlike_token()
    return dictionary[word]


def make_plural(word):
    for v1, v2 in zip("aeiou", "eioua"):
        if v1 in word:
            return word.replace(v1, v2, 1)
    return word + "e"


def add_article(word, plural=False):
    return word + ("u" if plural else "en")


def conjugate_verb(verb, tense="present"):
    base = get_base_word(verb)
    if tense == "past":
        return "na-" + base
    if tense == "future":
        return "ka-" + base
    return "la-" + base


def is_plural_noun(token):
    return token is not None and token.tag_ in ("NNS", "NNPS")


def noun_base_form(token):
    if token is None:
        return ""
    if token.pos_ in ("NOUN", "PROPN") and token.lemma_:
        return token.lemma_.lower()
    return token.text.lower()


def build_noun_phrase(token, adjectives, plural_nouns):
    if token is None:
        return ""

    base_english = noun_base_form(token)
    singular_form = get_base_word(base_english)
    plural = is_plural_noun(token)
    noun_form = make_plural(singular_form) if plural else singular_form
    phrase = add_article(noun_form, plural=plural)

    for adj in adjectives.get(token.i, []):
        phrase = get_base_word(adj) + " " + phrase

    if plural:
        plural_nouns.append(
            {
                "english": base_english,
                "singular": singular_form,
                "plural": noun_form,
            }
        )

    return phrase


def translate_clause(clause, return_metadata=False):
    doc = nlp(clause)

    verb = None
    subject = None
    obj = None
    adjectives = {}
    negative = False
    question = False
    tense = "present"
    copula_adj = None
    plural_nouns = []

    if clause.strip().endswith("?"):
        question = True

    for token in doc:
        if token.dep_ == "neg":
            negative = True

        if token.dep_ in ("nsubj", "nsubjpass") and subject is None:
            subject = token

        if token.dep_ in ("dobj", "pobj") and obj is None:
            obj = token

        if token.dep_ == "acomp":
            copula_adj = token.text

        if token.pos_ in ("VERB", "AUX") and verb is None:
            verb = token.lemma_
            if token.tag_ in ("VBD", "VBN"):
                tense = "past"
            elif token.tag_ in ("VB", "VBP", "VBZ"):
                tense = "present"
            elif token.tag_ == "MD":
                tense = "future"

        if token.pos_ == "ADJ" and token.head.pos_ in ("NOUN", "PROPN"):
            adjectives.setdefault(token.head.i, []).append(token.text)

    if subject is None:
        for token in doc:
            if token.pos_ in ("PRON", "NOUN", "PROPN"):
                subject = token
                break

    subject_word = build_noun_phrase(subject, adjectives, plural_nouns) if subject else ""

    if verb == "be" and copula_adj:
        adj_word = get_base_word(copula_adj)
        sentence_isl = f"{subject_word} {adj_word}".strip()
        if negative:
            sentence_isl = "no " + sentence_isl
        if question:
            sentence_isl += " va"
        if return_metadata:
            return sentence_isl, {"plural_nouns": plural_nouns}
        return sentence_isl

    object_word = build_noun_phrase(obj, adjectives, plural_nouns) if obj else ""
    verb_word = conjugate_verb(verb, tense) if verb else ""

    sentence_parts = [subject_word, verb_word]
    if object_word:
        sentence_parts.append(object_word)

    sentence_isl = " ".join(sentence_parts).strip()
    if negative:
        sentence_isl = "no " + sentence_isl
    if question:
        sentence_isl += " va"

    if return_metadata:
        return sentence_isl, {"plural_nouns": plural_nouns}
    return sentence_isl


def split_clauses(sentence):
    doc = nlp(sentence)
    clauses = []
    current_clause = []
    pending_conj = None

    for token in doc:
        if token.pos_ in ("CCONJ", "SCONJ"):
            if current_clause:
                clauses.append((" ".join(current_clause), pending_conj))
                current_clause = []
            pending_conj = CONJUNCTIONS.get(token.text.lower(), token.text.lower())
        elif not token.is_punct:
            current_clause.append(token.text)

    if current_clause:
        clauses.append((" ".join(current_clause), pending_conj))

    return clauses


def translate_sentence(sentence, return_metadata=False):
    clauses = split_clauses(sentence)
    isl_parts = []
    plural_nouns = []

    for i, (clause, conj) in enumerate(clauses):
        translated, metadata = translate_clause(clause, return_metadata=True)
        plural_nouns.extend(metadata["plural_nouns"])

        if i == 0:
            isl_parts.append(translated)
        else:
            if conj:
                isl_parts.append(conj)
            isl_parts.append(translated)

    sentence_isl = " ".join(isl_parts)
    if return_metadata:
        return sentence_isl, {"plural_nouns": plural_nouns}
    return sentence_isl


def pluralize_en(noun):
    return p.plural(noun)


def conjugate_en(verb, tense):
    if tense == "past":
        return verb + "ed"
    if tense == "future":
        return "will " + verb
    return verb + "s"


def sample_task_vocab(subject_count=12, object_count=12, verb_count=10, adjective_count=10):
    return {
        "subjects": random.sample(SUBJECTS, min(subject_count, len(SUBJECTS))),
        "objects": random.sample(OBJECTS, min(object_count, len(OBJECTS))),
        "verbs": random.sample(VERBS, min(verb_count, len(VERBS))),
        "adjectives": random.sample(ADJECTIVES, min(adjective_count, len(ADJECTIVES))),
    }


def random_clause(task_vocab):
    subj = random.choice(task_vocab["subjects"])
    obj = random.choice(task_vocab["objects"])
    verb = random.choice(task_vocab["verbs"])
    tense = random.choice(["past", "present", "future"])

    if random.random() < 0.5:
        subj = pluralize_en(subj)
    if random.random() < 0.5:
        obj = pluralize_en(obj)

    adj_subj = random.choice(task_vocab["adjectives"]) if random.random() < 0.5 else ""
    adj_obj = random.choice(task_vocab["adjectives"]) if random.random() < 0.5 else ""
    verb_form = conjugate_en(verb, tense)

    return (
        f"The {adj_subj + ' ' if adj_subj else ''}{subj} "
        f"{verb_form} the {adj_obj + ' ' if adj_obj else ''}{obj}"
    )


def random_sentence(task_vocab):
    sentence = random_clause(task_vocab)
    num_additional = 1

    for _ in range(num_additional):
        conj = random.choice(["and", "because", "but", "or"])
        sentence += f" {conj} {random_clause(task_vocab)}"

    return sentence


def detect_vowel_shift(singular, plural):
    for v1, v2 in zip("aeiou", "eioua"):
        if v1 in singular and singular.replace(v1, v2, 1) == plural:
            return (v1, v2)
    return None


def extract_plural_mappings(translation_metadata):
    mappings = set()
    for noun in translation_metadata.get("plural_nouns", []):
        shift = detect_vowel_shift(noun["singular"], noun["plural"])
        if shift:
            mappings.add(shift)
    return mappings


def plural_rules_covered(example_translations, test_translation):
    example_mappings = set()
    for translation in example_translations:
        example_mappings |= extract_plural_mappings(translation)
    test_mappings = extract_plural_mappings(test_translation)
    return test_mappings.issubset(example_mappings)


def extract_features_en(sentence):
    doc = nlp(sentence)
    features = {
        "tenses": set(),
        "negation": False,
        "copula": False,
        "adjectives": False,
        "conjunctions": set(),
        "multi_clause": False,
    }
    clause_count = 1

    for token in doc:
        if token.tag_ in ("VBD", "VBN"):
            features["tenses"].add("past")
        elif token.tag_ in ("VBZ", "VBP"):
            features["tenses"].add("present")
        elif token.text.lower() == "will":
            features["tenses"].add("future")

        if token.dep_ == "neg":
            features["negation"] = True
        if token.lemma_ == "be":
            features["copula"] = True
        if token.pos_ == "ADJ":
            features["adjectives"] = True
        if token.pos_ in ("CCONJ", "SCONJ"):
            features["conjunctions"].add(token.text.lower())
            clause_count += 1

    if clause_count > 1:
        features["multi_clause"] = True

    return features


def features_covered(example_sentences, test_sentence):
    test_features = extract_features_en(test_sentence)
    combined = {
        "tenses": set(),
        "negation": False,
        "copula": False,
        "adjectives": False,
        "conjunctions": set(),
        "multi_clause": False,
    }

    for example in example_sentences:
        features = extract_features_en(example)
        combined["tenses"] |= features["tenses"]
        combined["conjunctions"] |= features["conjunctions"]
        if features["negation"]:
            combined["negation"] = True
        if features["copula"]:
            combined["copula"] = True
        if features["adjectives"]:
            combined["adjectives"] = True
        if features["multi_clause"]:
            combined["multi_clause"] = True

    if not test_features["tenses"].issubset(combined["tenses"]):
        return False
    if test_features["negation"] and not combined["negation"]:
        return False
    if test_features["copula"] and not combined["copula"]:
        return False
    if test_features["adjectives"] and not combined["adjectives"]:
        return False
    if not test_features["conjunctions"].issubset(combined["conjunctions"]):
        return False
    if test_features["multi_clause"] and not combined["multi_clause"]:
        return False

    return True


def extract_base_words_en(sentence):
    base_words = set()
    for token in nlp(sentence):
        if not token.is_alpha:
            continue
        if token.pos_ in ("NOUN", "PROPN"):
            base_words.add(noun_base_form(token))
            continue
        if token.lemma_:
            base_words.add(token.lemma_.lower())
        else:
            base_words.add(token.text.lower())
    return base_words


def test_words_covered(example_sentences, test_sentence):
    example_base_words = set()
    for sentence in example_sentences:
        example_base_words.update(extract_base_words_en(sentence))

    test_base_words = extract_base_words_en(test_sentence)
    return test_base_words.issubset(example_base_words)


def task_is_valid(examples_en, examples_el, test_en, test_el):
    if not test_words_covered(examples_en, test_en):
        return False

    if not features_covered(examples_en, test_en):
        return False

    example_metadata = [metadata for _, metadata in examples_el]
    test_metadata = test_el[1]
    if not plural_rules_covered(example_metadata, test_metadata):
        return False

    return True


def generate_task(task_id):
    while True:
        task_vocab = sample_task_vocab()
        examples_en = [random_sentence(task_vocab) for _ in range(5)]
        examples_el = [translate_sentence(sentence, return_metadata=True) for sentence in examples_en]
        test_en = random_sentence(task_vocab)
        test_el = translate_sentence(test_en, return_metadata=True)

        if task_is_valid(examples_en, examples_el, test_en, test_el):
            return {
                "id": task_id,
                "vocab": task_vocab,
                "examples": [{"en": en, "el": el[0]} for en, el in zip(examples_en, examples_el)],
                "test": {"en": test_en, "el": test_el[0]},
            }


if __name__ == "__main__":
    n_tasks = 100
    output_path = "isl_dataset2.json"

    dataset = {"tasks": [generate_task(i + 1) for i in trange(n_tasks)]}
    export_data = [
        {
            "train": [[example["en"], example["el"]] for example in task["examples"]],
            "test": [[task["test"]["en"], task["test"]["el"]]],
        }
        for task in dataset["tasks"]
    ]
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(export_data, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset['tasks'])} tasks to {output_path}")
