"""Compile an accepted rule set into a standalone sequence generator."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .automaton import RuleAutomaton
from .rules import CdfSampler, Rule, ruleset_entropy, validate_probs


@dataclass(frozen=True)
class CompiledGenerator:
    """Sampler for a fixed rule set; the automaton is built once."""
    base_probs: np.ndarray
    rules: List[Rule]
    automaton: RuleAutomaton
    base_sampler: CdfSampler
    vocab_size: int
    precompute_transitions: bool

    def generate(
        self,
        num_sims: int = 1,
        sim_len: int = 1000,
        seed: Optional[int] = None,
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None,
    ) -> np.ndarray:
        """Return an array of shape (num_sims, sim_len), or (sim_len,) when num_sims == 1."""
        num_sims = int(num_sims)
        sim_len = int(sim_len)
        if num_sims <= 0 or sim_len <= 0:
            raise ValueError("num_sims and sim_len must be positive")

        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(num_sims)

        def _run_one(child_seed: np.random.SeedSequence) -> np.ndarray:
            rng = np.random.default_rng(child_seed)
            seq = np.empty(sim_len, dtype=int)
            state = 0
            for t in range(sim_len):
                ridx = self.automaton.best_rule_idx[state]
                if ridx == -1:
                    sym = self.base_sampler.pick(rng)
                else:
                    out = self.rules[ridx].try_apply(rng)
                    if out is None:
                        sym = self.base_sampler.pick(rng)
                    else:
                        sym = int(out)

                seq[t] = sym
                state = self.automaton.step_state(state, sym)
            return seq

        total_steps = num_sims * sim_len
        if parallel is None:
            # Threads only pay off for large workloads.
            parallel = (num_sims >= 8 and total_steps >= 200_000)

        if parallel:
            workers = max_workers or min(num_sims, (os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                seqs = list(ex.map(_run_one, child_seeds))
        else:
            seqs = [_run_one(s) for s in child_seeds]

        arr = np.stack(seqs, axis=0)
        if num_sims == 1:
            return arr[0]
        return arr

    @property
    def ruleset_entropy(self) -> float:
        """Reliability-weighted entropy (nats) of the rule output distributions."""
        return ruleset_entropy(self.rules, weighted_by_reliability=True)


def compile_generator(
    base_probs: np.ndarray,
    rules: List[Rule],
    precompute_transitions: bool = True,
) -> CompiledGenerator:
    base_probs = validate_probs(np.asarray(base_probs, dtype=float), "base_probs")
    V = len(base_probs)
    autom = RuleAutomaton(rules, alphabet_size=V, precompute_transitions=precompute_transitions)
    base_sampler = CdfSampler(base_probs)
    return CompiledGenerator(
        base_probs=base_probs,
        rules=rules,
        automaton=autom,
        base_sampler=base_sampler,
        vocab_size=V,
        precompute_transitions=precompute_transitions,
    )
