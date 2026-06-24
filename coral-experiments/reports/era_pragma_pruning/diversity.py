"""How diverse are singleton rankings across each dataset?

Three measures per qid:
  - n_answers: size of the answer space
  - n_distinct_singletons: how many distinct full singleton rankings exist
  - max_class_share: fraction of pragmas/criteria in the plurality class
  - top1_agreement: fraction of singletons whose top-1 matches the baseline top-1
"""
from collections import Counter
from statistics import mean, median

from search.lattice import combo_key


def analyze(name: str, lats: dict[str, tuple[list[str], dict[str, list[str]]]]) -> None:
    n_answers_list = []
    n_distinct_list = []
    plurality_share = []
    top1_agree = []

    for qid, (crit, rbc) in lats.items():
        baseline = rbc[combo_key(crit)]
        if not baseline:
            continue
        baseline_top1 = baseline[0]
        n_answers = len(baseline)
        n_answers_list.append(n_answers)

        singletons = [tuple(rbc[c]) for c in crit if c in rbc]
        cnt = Counter(singletons)
        n_distinct = len(cnt)
        max_share = cnt.most_common(1)[0][1] / len(singletons)
        agree = sum(1 for r in singletons if r and r[0] == baseline_top1) / len(singletons)

        n_distinct_list.append(n_distinct)
        plurality_share.append(max_share)
        top1_agree.append(agree)

    print(f"\n  {name}  (n_qids={len(lats)})")
    print(f"    answer-space size       median = {median(n_answers_list):.1f}")
    print(f"    distinct singletons     median = {median(n_distinct_list):.1f}   /  n_criteria")
    print(f"    distinct / n_criteria   median = {median(d/10 for d in n_distinct_list):.2f}")
    print(f"    plurality class share   median = {median(plurality_share):.2f}   mean = {mean(plurality_share):.2f}")
    print(f"    top1 == baseline top1   median = {median(top1_agree):.2f}   mean = {mean(top1_agree):.2f}")


from scripts.msmarco.run_subsample import _load_all_qids as m_q, _load_lattices as m_l
analyze("MSMARCO (criteria)", m_l(m_q()))

from scripts.sonnet.run_subsample import _load_all_qids as s_q, _load_lattices as s_l
analyze("SONNET  (criteria)", s_l(s_q()))

from scripts.pragma.msmarco.run_subsample import _load_all_qids as pm_q, _load_lattices as pm_l
analyze("MSMARCO (pragma)",   pm_l(pm_q()))

from scripts.pragma.sonnet.run_subsample import _load_all_qids as ps_q, _load_pragma_lattices as ps_l
analyze("SONNET  (pragma)",   ps_l(ps_q()))
