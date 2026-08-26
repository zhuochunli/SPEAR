import re
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
import spacy
from utils import verify_preds


# Note: You must run `python -m spacy download en_core_web_sm` first.
nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])

MAX_ANCHORS = 100  # Bounds LCS complexity to O(100x100) per sample


def _per_sample(value, count):
    """Broadcast scalar metadata or validate per-completion batch metadata."""
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"Expected {count} metadata values, got {len(value)}.")
        return value
    return [value] * count


def format_reward(completions, **kwargs):
    """
    Checks whether the model output follows the required structured format:
        <think>...</think><answer>...</answer>
    Returns 1.0 if valid, 0.0 otherwise.
    """
    rewards = []
    for comp in completions:
        text = comp[0]["content"].strip()
        if (text.count("<think>") == 1 and text.count("</think>") == 1 and
            text.count("<answer>") == 1 and text.count("</answer>") == 1 and
            text.find("<think>") < text.find("</think>") < text.find("<answer>") < text.find("</answer>")):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions, **kwargs):
    """
    Extracts the answer from the <answer> tag and verifies it against the
    gold solution using task-appropriate matching (math_verify for math,
    string matching for QA tasks).
    Returns 1.0 if correct, 0.0 otherwise.
    """
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    task_types = _per_sample(kwargs.get("task_type", "gsm8k"), len(completion_contents))
    rewards = []

    for content, solution, task_type in zip(completion_contents, solutions, task_types):
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if not answer_match:
            rewards.append(0.0)
            continue
        answer_str = answer_match.group(1).strip()
        rewards.append(1.0 if verify_preds(answer_str, solution, task_type) else 0.0)

    return rewards


def get_lcs_length(seq1, seq2):
    """
    Computes the Longest Common Subsequence length between two anchor sequences
    to measure order-aware reasoning alignment.
    Time complexity: O(|seq1| x |seq2|), bounded by MAX_ANCHORS^2.
    """
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def get_lcs_f1_score(teacher_seq, student_seq):
    """Return the LCS-F1 score for two non-token reasoning trajectories."""
    if not teacher_seq or not student_seq:
        return 0.0
    lcs_len = get_lcs_length(teacher_seq, student_seq)
    precision = lcs_len / len(student_seq)
    recall = lcs_len / len(teacher_seq)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


MATH_OPS = re.compile(
    r'\b(multiply|divide|subtract|add|sum|total|average|percent|'
    r'ratio|compute|calculate|simplify|solve|equal|let|assume|'
    r'first|next|then|finally|therefore|because|since|thus)\b',
    re.IGNORECASE
)

LOGIC_OPS = re.compile(
    r'\b(because|therefore|since|thus|hence|if|then|so|'
    r'however|although|unless|implies|means|conclude)\b',
    re.IGNORECASE
)

# def extract_anchors(content, task_type):
#     """
#     Extracts ordered reasoning anchors that are style-independent and
#     answer-value-independent, so LCS measures HOW the model reasons
#     rather than WHAT values it computes.

#     - math/gsm8k    : operation verbs + logical connectives
#                       (avoids numbers/values which correlate with final answer)
#     - gpqa          : causal/scientific verbs + logical connectives
#                       (avoids domain nouns which mismatch across model sizes)
#     - commonsenseQA : verb lemmas + logical connectives
#                       (lemmatized to handle paraphrase variation)
#     """
#     if not content:
#         return []

#     if task_type in ["math", "gsm8k"]:
#         ops = MATH_OPS.findall(content.lower())
#         var_names = re.findall(r'\b([a-zA-Z])\s*=', content)  # structure only, not values
#         anchors = ops + var_names

#     elif task_type == "gpqa":
#         doc = nlp(content)
#         anchors = []
#         for token in doc:
#             # Causal verbs fire regardless of domain vocabulary
#             if token.lemma_ in ["cause", "increase", "decrease", "inhibit",
#                                   "activate", "require", "produce", "result",
#                                   "lead", "prevent", "affect", "depend",
#                                   "bind", "block", "encode", "express"]:
#                 anchors.append(token.lemma_)
#         anchors += LOGIC_OPS.findall(content.lower())

#     else:  # commonsenseQA
#         doc = nlp(content)
#         anchors = []
#         for token in doc:
#             if token.pos_ == "VERB" and not token.is_stop and len(token.text) > 2:
#                 anchors.append(token.lemma_)  # lemmatize — catches paraphrases
#         anchors += LOGIC_OPS.findall(content.lower())

#     return anchors[:MAX_ANCHORS]

def get_span(token):
    """
    Returns a short, stable noun phrase by collecting only compound
    modifiers of the token. Avoids subtree explosion while capturing
    meaningful multi-word reagent/entity names.
    """
    compounds = [t.text for t in token.lefts if t.dep_ == "compound"]
    return " ".join(compounds + [token.text]).lower()

def extract_anchors(content, task_type):
    """
    Extracts a domain-specific ordered sequence of reasoning anchors from text.

    - math/gsm8k : LaTeX expressions and variable assignments
                   (bare numbers excluded — too noisy for LCS alignment)
    - gpqa        : Verb-Object relational triples from NER + dependency parse
    - others      : Noun chunk roots and their governing verbs
    Sequences are truncated to MAX_ANCHORS to keep LCS tractable.
    """
    if not content:
        return []
    anchors = []

    # --- MATH & GSM8K ---
    if task_type in ["math", "gsm8k"]:
        found = re.findall(r'\$.*?\$|\\\[.*?\\\]|[a-zA-Z]\s*=\s*-?\d+\.?\d*', content)
        anchors = [str(item).strip() for item in found]

    # --- GPQA (Science) ---
    # elif task_type == "gpqa":
    #     doc = nlp(content)
    #     for token in doc:
    #         if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
    #             if token.dep_ in ["nsubj", "dobj"] and token.head.pos_ == "VERB":
    #                 # Relational form only — avoids double-counting bare noun + pair
    #                 anchors.append(f"{token.text.lower()}_{token.head.lemma_}")
    #             else:
    #                 anchors.append(token.text.lower())
    elif task_type == "gpqa":
        doc = nlp(content)
        seen = set()

        for token in doc:
            if token.pos_ != "VERB":
                continue

            subj = next(
                (t for t in token.lefts  if t.dep_ in ("nsubj", "nsubjpass")),
                None
            )
            obj  = next(
                (t for t in token.rights if t.dep_ in ("dobj", "obj", "attr")),
                None
            )

            anchor = None

            if obj:
                # Primary: verb + obj
                # In chemistry, the action and its target are more invariant
                # than who performs it — "adds a methyl group" is the core
                # fact regardless of whether the subject is "MeMgBr",
                # "the Grignard reagent", or "the organometallic compound"
                anchor = f"{token.lemma_} {get_span(obj)}"
            elif subj and subj.pos_ in ("NOUN", "PROPN"):
                # Fallback: subj + verb
                # For intransitive verbs (e.g. "the count increases"),
                # the subject carries the core semantic signal
                anchor = f"{get_span(subj)} {token.lemma_}"

            if anchor and anchor not in seen and 2 <= len(anchor.split()) <= 4:
                seen.add(anchor)
                anchors.append(anchor)

    # --- Commonsense / General ---
    else:
        doc = nlp(content)
        for chunk in doc.noun_chunks:
            if len(chunk.root.text) > 2:
                anchors.append(chunk.root.text.lower())
                if chunk.root.head.pos_ == "VERB":
                    anchors.append(chunk.root.head.lemma_)

    return anchors[:MAX_ANCHORS]


def reasoning_reward(completions, **kwargs):
    """
    Order-Aware Reasoning Alignment Reward via LCS-F1.

    Compares the student's <think> reasoning to the teacher's rationale using
    LCS over domain-specific anchor sequences. F1 scoring balances:
        - Recall:    how well the student covers the teacher's reasoning milestones
        - Precision: penalizes bloated/padded reasoning that inflates recall alone

    Note: LCS acts as a soft constraint — students are not penalized for valid
    alternative reasoning paths that share key milestones with the teacher.
    Returns a score in [0.0, 1.0].
    """
    teacher_thinks = kwargs["deepseek_rationale"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    task_types = _per_sample(kwargs.get("task_type", "gsm8k"), len(completion_contents))

    for content, teacher_think, task_type in zip(completion_contents, teacher_thinks, task_types):
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        student_think = match.group(1).strip()

        t_seq = extract_anchors(teacher_think, task_type)
        s_seq = extract_anchors(student_think, task_type)

        # No anchors in teacher rationale → no reliable signal, skip sample
        if not t_seq or not s_seq:
            rewards.append(0.0)
            continue

        # LCS-F1 penalizes both missing teacher milestones (recall) and
        # unsupported or repetitive student milestones (precision).
        rewards.append(get_lcs_f1_score(t_seq, s_seq))

    return rewards


def combined_reward(completions, **kwargs):
    """
    Master reward function combining format gating, accuracy, and reasoning alignment.

    Reward structure:
        - Format: hard gate — malformed outputs receive zero reward.
        - Accuracy (R_acc):        primary signal,   weight = 1.0  → range [0, 1]
        - Reasoning (R_reasoning): secondary signal, weight = 0.5 → range [0, 0.5]

    Total range: [0.0, 1.5]
        - Max (1.5): correct answer + perfect reasoning alignment
        - Min (0.0): wrong format + no reasoning signal
    """
    fmt_rewards       = format_reward(completions, **kwargs)
    acc_rewards       = accuracy_reward(completions, **kwargs)
    reasoning_rewards = reasoning_reward(completions, **kwargs)

    final_rewards = []
    for fmt, acc, reasoning in zip(fmt_rewards, acc_rewards, reasoning_rewards):
        if fmt < 1.0:
            final_rewards.append(0.0)
            continue

        reward = (1.0 * acc) + (0.5 * reasoning)
        final_rewards.append(reward)

    return final_rewards
