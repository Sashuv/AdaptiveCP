"""
Dataset loaders for TriviaQA, WebQuestions, and MMLU.
"""

import random
from datasets import load_dataset

from core import QASample, MMLU_SUBJECTS, assign_split, print_split_counts


def load_triviaqa(seed: int, n_total: int = 3000) -> list:
    raw = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    raw = raw.shuffle(seed=seed).select(range(min(n_total, len(raw))))

    samples = []
    for i, row in enumerate(raw):
        aliases = row["answer"]["aliases"] or [row["answer"]["value"]]
        samples.append(QASample(
            question_id  = f"triviaqa_{i}",
            question     = row["question"],
            gold_answers = [a.strip().lower() for a in aliases],
            source       = "triviaqa",
            split        = assign_split(i, n_total),
        ))

    print(f"TriviaQA loaded: {len(samples)} samples")
    print_split_counts(samples)
    return samples


def load_webquestions(seed: int) -> list:
    raw = load_dataset("stanfordnlp/web_questions", split="train")
    raw = raw.shuffle(seed=seed)
    n_total = len(raw)

    samples = []
    for i, row in enumerate(raw):
        answers = [a.strip().lower() for a in row["answers"]]
        samples.append(QASample(
            question_id  = f"webq_{i}",
            question     = row["question"],
            gold_answers = answers,
            source       = "webq",
            split        = assign_split(i, n_total),
        ))

    print(f"WebQuestions loaded: {len(samples)} samples")
    print_split_counts(samples)
    return samples


def load_mmlu(seed: int, subjects: list = MMLU_SUBJECTS) -> list:
    """
    MMLU multiple-choice → open-ended format.
    gold_answers = [correct_choice_text] (not the letter).
    """
    LETTERS = ["A", "B", "C", "D"]
    all_samples = []

    for subject in subjects:
        raw = load_dataset("cais/mmlu", subject, split="test")
        for i, row in enumerate(raw):
            choices    = row["choices"]
            answer_idx = row["answer"]
            correct    = choices[answer_idx].strip().lower()

            choices_text = "\n".join(
                f"{LETTERS[j]}) {choices[j]}" for j in range(len(choices))
            )
            question = f"{row['question']}\n{choices_text}"

            all_samples.append(QASample(
                question_id  = f"mmlu_{subject}_{i}",
                question     = question,
                gold_answers = [correct],
                source       = "mmlu",
                split        = "",
            ))

    random.Random(seed).shuffle(all_samples)
    n_total = len(all_samples)
    for i, s in enumerate(all_samples):
        s.split = assign_split(i, n_total)

    print(f"MMLU loaded: {n_total} samples across {len(subjects)} subjects")
    print_split_counts(all_samples)
    return all_samples
