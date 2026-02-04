# src/utils/postprocess.py
import math
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch

NEG_INF = -1e30

def _log_add_exp(a: float, b: float) -> float:
    if a <= NEG_INF: return b
    if b <= NEG_INF: return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def _ctc_prefix_beam_search(logp: np.ndarray, beam_width: int, blank: int = 0):
    # logp: [T, C] in log domain
    T, C = logp.shape
    beams = {(): (0.0, NEG_INF)}  # prefix -> (p_blank, p_nonblank)

    for t in range(T):
        next_beams = {}
        for prefix, (pb, pnb) in beams.items():
            for c in range(C):
                p = float(logp[t, c])

                if c == blank:
                    nb_pb, nb_pnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    nb_pb = _log_add_exp(nb_pb, pb + p)
                    nb_pb = _log_add_exp(nb_pb, pnb + p)
                    next_beams[prefix] = (nb_pb, nb_pnb)
                    continue

                last = prefix[-1] if prefix else None

                # Case 1: extend with same char
                if c == last:
                    # stay on same prefix from non-blank
                    nb_pb, nb_pnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    nb_pnb = _log_add_exp(nb_pnb, pnb + p)
                    next_beams[prefix] = (nb_pb, nb_pnb)

                    # extend prefix from blank (creates repeated char)
                    new_prefix = prefix + (c,)
                    nb_pb, nb_pnb = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    nb_pnb = _log_add_exp(nb_pnb, pb + p)
                    next_beams[new_prefix] = (nb_pb, nb_pnb)
                else:
                    new_prefix = prefix + (c,)
                    nb_pb, nb_pnb = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    nb_pnb = _log_add_exp(nb_pnb, pb + p)
                    nb_pnb = _log_add_exp(nb_pnb, pnb + p)
                    next_beams[new_prefix] = (nb_pb, nb_pnb)

        # prune
        scored = []
        for pref, (pb, pnb) in next_beams.items():
            total = _log_add_exp(pb, pnb)
            scored.append((total, pref, pb, pnb))
        scored.sort(key=lambda x: x[0], reverse=True)
        beams = {pref: (pb, pnb) for (total, pref, pb, pnb) in scored[:beam_width]}

    # return list sorted
    finals = []
    for pref, (pb, pnb) in beams.items():
        total = _log_add_exp(pb, pnb)
        finals.append((total, pref))
    finals.sort(key=lambda x: x[0], reverse=True)
    return finals  # [(logprob, prefix_tuple), ...]

def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 1,
    conf_mode: str = "meanmax"
) -> List[Tuple[str, float]]:
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)

    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    logp_np = preds.detach().cpu().numpy()  # log-probs [B,T,C]

    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []

    for b in range(batch_size):
        if beam_width <= 1:
            path = indices_np[b]
            probs_b = max_probs_np[b]

            pred_chars = []
            confidences = []
            time_idx = 0
            for char_idx, group in groupby(path):
                group_list = list(group)
                group_size = len(group_list)
                if char_idx != 0:
                    # AFTER
                    k = int(char_idx)
                    pred_chars.append(idx2char.get(k, idx2char.get(str(k), '')))
                    group_probs = probs_b[time_idx:time_idx + group_size]
                    confidences.append(float(np.max(group_probs)))
                time_idx += group_size

            pred_str = "".join(pred_chars)

            if not confidences:
                conf = 0.0
            else:
                if conf_mode == "geom":
                    eps = 1e-9
                    conf = float(np.exp(np.mean(np.log(np.array(confidences) + eps))))
                else:  # meanmax
                    conf = float(np.mean(confidences))

            results.append((pred_str, conf))
        else:
            beams = _ctc_prefix_beam_search(logp_np[b], beam_width=beam_width, blank=0)
            best_logp, best_pref = beams[0]
            pred_str = "".join(idx2char.get(int(c), "") for c in best_pref)

            # Confidence from margin between top1 & top2 (best for tie-break)
            if conf_mode == "margin" and len(beams) > 1:
                margin = best_logp - beams[1][0]
                conf = 1.0 / (1.0 + math.exp(-margin))
            else:
                # fallback: length-normalized probability
                L = max(1, len(pred_str))
                conf = float(math.exp(best_logp / L))
                conf = max(0.0, min(1.0, conf))

            results.append((pred_str, conf))

    return results