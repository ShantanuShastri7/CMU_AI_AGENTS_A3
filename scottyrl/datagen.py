"""Trajectory -> training data. *** STUDENT IMPLEMENTATION FILE ***

The heart of the data pipeline: turning a multi-turn episode into stitched
token sequences with correct masking, logprob alignment, and advantage
broadcasting. Read the TokenSequence docstring in `scottyrl/types.py`
carefully before starting, especially the off-by-one: position j of
mask/logprobs/advantages describes the *target* token tokens[j+1].

Implement the three functions below. You may add private helpers in this
file, but these signatures must not change.
"""

from __future__ import annotations

from scottyrl.types import TokenSequence, Trajectory


def extension_suffix(accumulated: list[int], prompt_tokens: list[int]) -> list[int] | None:
    """Check the sequence-extension property between an accumulated token
    sequence and the next call's prompt.

    Contract: if `accumulated` is a (non-strict) prefix of `prompt_tokens`,
    return the remaining suffix of `prompt_tokens` (the tokens of the newly
    rendered tail: turn markers, environment observation, generation prompt;
    may be empty). Otherwise return None: the extension is broken and the
    caller must start a new sequence.

    Examples:
        extension_suffix([1, 2, 3], [1, 2, 3, 9, 9]) -> [9, 9]
        extension_suffix([1, 2, 3], [1, 2, 3])       -> []
        extension_suffix([1, 2, 3], [1, 9, 3, 4])    -> None
        extension_suffix([1, 2, 3], [1, 2])          -> None
        extension_suffix([], [1, 2])                 -> [1, 2]  (an empty sequence is a prefix of anything)
    """
    if not accumulated:
        return prompt_tokens[:]
    if len(prompt_tokens) < len(accumulated):
        return None
    if prompt_tokens[:len(accumulated)] != accumulated:
        return None
    return prompt_tokens[len(accumulated):]


def _get_call_fields(call):
    """Robustly extract prompt, completion, and logprobs from an LLM call object."""
    prompt = getattr(call, "prompt_tokens", getattr(call, "prompt", None))
    completion = getattr(call, "completion_tokens", getattr(call, "completion", None))
    logprobs = getattr(call, "logprobs", getattr(call, "completion_logprobs", None))
    return prompt, completion, logprobs


def stitch_trajectory(traj: Trajectory) -> list[TokenSequence]:
    """Stitch an episode's LLM calls into training sequences.

    Output contract:
      - Walk the calls in order, maintaining one growing token sequence of
        context tokens (prompt + rendered env observations) interleaved with
        completion tokens. A call whose prompt extends the sequence built so
        far continues it: only the new context suffix is appended before the
        call's completion. A call that breaks the extension property ends the
        current sequence and starts a new one from that call's full prompt.
      - In every emitted TokenSequence: mask is 1 at exactly the target
        positions whose token came from a completion; logprobs carries the
        sampling logprob at those positions and 0.0 everywhere else;
        advantages is all zeros at this stage.
      - Never emit a sequence with fewer than 2 tokens or with no action
        (mask == 1) positions; there would be nothing to train on.

    Example -- two calls whose prompts extend:
        call 0: prompt=[10, 11], completion=[20, 21], logprobs=[-0.2, -0.3]
        call 1: prompt=[10, 11, 20, 21, 30], completion=[40], logprobs=[-0.4]

    Before call 1, the accumulated sequence is [10, 11, 20, 21], so only
    prompt suffix [30] is appended. The single result is:

        tokens     = [10, 11, 20, 21, 30, 40]
        targets    =      [11, 20, 21, 30, 40]  (= tokens[1:])
        mask       =      [ 0,  1,  1,  0,  1]
        logprobs   =      [0.0, -0.2, -0.3, 0.0, -0.4]
        advantages =      [0.0, 0.0, 0.0, 0.0, 0.0]

    Example -- the second prompt breaks the extension property:
        call 0: prompt=[10, 11], completion=[20, 21]
        call 1: prompt=[10, 11, 30], completion=[40]

    Since [10, 11, 20, 21] is not a prefix of call 1's prompt, this emits two
    results. Their tokens and masks are:

        tokens=[10, 11, 20, 21], mask=[0, 1, 1]
        tokens=[10, 11, 30, 40], mask=[0, 0, 1]

    Returns: list of TokenSequence (possibly empty).
    """
    sequences: list[TokenSequence] = []

    acc_tokens: list[int] = []
    acc_mask: list[int] = []
    acc_logprobs: list[float] = []
    acc_advantages: list[float] = []

    def flush_current():
        nonlocal acc_tokens, acc_mask, acc_logprobs, acc_advantages
        if len(acc_tokens) >= 2 and any(m == 1 for m in acc_mask):
            sequences.append(
                TokenSequence(
                    tokens=list(acc_tokens),
                    mask=list(acc_mask),
                    logprobs=list(acc_logprobs),
                    advantages=list(acc_advantages),
                )
            )
        acc_tokens = []
        acc_mask = []
        acc_logprobs = []
        acc_advantages = []

    for call in traj.calls:
        prompt, completion, logprobs = _get_call_fields(call)

        # Check if call's prompt extends the currently accumulated token sequence
        is_extension = (
            len(acc_tokens) > 0
            and len(prompt) >= len(acc_tokens)
            and prompt[: len(acc_tokens)] == acc_tokens
        )

        if not is_extension:
            # Extension broken: emit current accumulated sequence and start a new one
            flush_current()

            if not prompt:
                continue

            # Initialize new sequence with prompt
            acc_tokens.append(prompt[0])
            for p_tok in prompt[1:]:
                acc_tokens.append(p_tok)
                acc_mask.append(0)
                acc_logprobs.append(0.0)
                acc_advantages.append(0.0)

            prompt_suffix = []
        else:
            # Extension succeeded: only take prompt tokens that weren't in acc_tokens yet
            prompt_suffix = prompt[len(acc_tokens) :]

        # Append prompt suffix tokens
        for p_tok in prompt_suffix:
            acc_tokens.append(p_tok)
            acc_mask.append(0)
            acc_logprobs.append(0.0)
            acc_advantages.append(0.0)

        # Append completion tokens and their sampling logprobs
        for c_tok, lp in zip(completion, logprobs):
            acc_tokens.append(c_tok)
            acc_mask.append(1)
            acc_logprobs.append(float(lp))
            acc_advantages.append(0.0)

    # Flush any remaining accumulated sequence at the end of the trajectory
    flush_current()

    return sequences


def trajectory_to_training_sequences(traj: Trajectory, advantage: float) -> list[TokenSequence]:
    """Stitch a trajectory and broadcast a scalar advantage over its action
    tokens: every position with mask == 1 gets `advantage`, every other
    position stays 0.0. (In REINFORCE/GRPO, all tokens the policy generated
    in an episode share the episode-level advantage.)
    """
    sequences = stitch_trajectory(traj)
    training_sequences: list[TokenSequence] = []

    for seq in sequences:
        # Broadcast `advantage` to positions where mask == 1, keep 0.0 for mask == 0
        broadcasted_advantages = [
            float(advantage) if m == 1 else 0.0
            for m in seq.mask
        ]

        # Construct new TokenSequence with updated advantages
        training_sequences.append(
            TokenSequence(
                tokens=seq.tokens,
                mask=seq.mask,
                logprobs=seq.logprobs,
                advantages=broadcasted_advantages,
            )
        )

    return training_sequences
