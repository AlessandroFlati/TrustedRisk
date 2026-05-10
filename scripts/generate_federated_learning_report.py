"""Phase 17.AN - Render the federated-learning report."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.federated_learning import (
    privacy_utility_curve, run_federated_training,
)


_SITE_IDS = [
    "site_urban_academic",
    "site_rural_community",
    "site_safety_net",
    "site_geriatric_specialty",
    "site_pediatric_adjacent",
]


def _render_md(rep, pu) -> str:
    lines = ["# TrustedRisk - Federated Learning report", ""]
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    lines.append(
        f"**Generated**: {timestamp} - "
        f"**Sites**: {rep.n_sites} - "
        f"**Rounds**: {rep.n_rounds} - "
        f"**Centralized baseline ECE**: "
        f"{rep.centralized_baseline_ece:.4f}"
    )
    lines.append("")
    lines.append(
        "Simulates McMahan 2017 FedAvg over five biased hospital "
        "sites with no raw-data sharing. Each round broadcasts only "
        "the Beta-Binomial parameters (alpha_post, beta_post per "
        "LACE bin)."
    )
    lines.append("")
    lines.append("## 1. Per-round convergence (no DP)")
    lines.append("")
    lines.append(
        "| round | global ECE | site_urban | site_rural | "
        "site_safety_net | site_geri | site_pedi |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in rep.rounds:
        s = r.per_site_ece
        lines.append(
            f"| {r.round_index} | {r.global_ece:.4f} | "
            f"{s.get('site_urban_academic', 0):.4f} | "
            f"{s.get('site_rural_community', 0):.4f} | "
            f"{s.get('site_safety_net', 0):.4f} | "
            f"{s.get('site_geriatric_specialty', 0):.4f} | "
            f"{s.get('site_pediatric_adjacent', 0):.4f} |"
        )
    lines.append("")
    lines.append("## 2. Local-only baseline ECE")
    lines.append("")
    lines.append("| site | local ECE |")
    lines.append("| --- | ---: |")
    for site, ece in sorted(rep.local_only_ece.items()):
        lines.append(f"| `{site}` | {ece:.4f} |")
    lines.append("")
    lines.append("## 3. Privacy-utility curve (DP-FedAvg)")
    lines.append("")
    lines.append(
        "| epsilon | final ECE | centralized | overhead |"
    )
    lines.append("| ---: | ---: | ---: | ---: |")
    for p in pu.points:
        lines.append(
            f"| {p.epsilon_per_round} | "
            f"{p.final_global_ece:.4f} | "
            f"{p.centralized_baseline_ece:.4f} | "
            f"{p.federated_overhead:+.4f} |"
        )
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append(
        "- McMahan et al. 2017 - Communication-Efficient Learning "
        "of Deep Networks from Decentralized Data (AISTATS).\n"
        "- Geyer et al. 2017 - Differentially Private Federated "
        "Learning.\n"
        "- Dwork et al. 2006 - Calibrating Noise to Sensitivity."
    )
    return "\n".join(lines)


def main() -> int:
    out_dir = ROOT / "docs" / "federated"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = run_federated_training(
        site_ids=_SITE_IDS,
        per_site_n=1000, n_rounds=10,
    )
    pu = privacy_utility_curve(
        site_ids=_SITE_IDS,
        per_site_n=1000, n_rounds=10,
        epsilon_grid=[0.1, 0.5, 1.0, 5.0, 10.0],
    )
    md = _render_md(rep, pu)
    (out_dir / "FEDERATED_LEARNING.md").write_text(
        md, encoding="utf-8")
    (out_dir / "federated_learning.json").write_text(
        json.dumps(rep.model_dump(mode="json"), indent=2),
        encoding="utf-8")
    (out_dir / "privacy_utility_curve.json").write_text(
        json.dumps(pu.model_dump(mode="json"), indent=2),
        encoding="utf-8")
    final_ece = rep.rounds[-1].global_ece
    print(
        f"FedAvg: final global ECE {final_ece:.4f} vs centralized "
        f"baseline {rep.centralized_baseline_ece:.4f} "
        f"(overhead {final_ece - rep.centralized_baseline_ece:+.4f})."
    )
    print(
        "Privacy-utility curve: lowest ECE "
        f"{min(p.final_global_ece for p in pu.points):.4f} at "
        f"epsilon {max(p.epsilon_per_round for p in pu.points)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
