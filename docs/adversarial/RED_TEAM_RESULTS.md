# Red-Team Results -- v2 Corpus

- Corpus: `redteam-v2-2026-04`
- Target: `detect_phi`
- Generated: 2026-04-30 05:47 UTC
- Cases: **110**
- Passed: **110** (rate 1.000)
- Failed: **0**
- Posture: **PASS**

## Pass-rate by category

| Category | Pass rate |
|---|---|
| bias_probe | 1.000 |
| citation_fabrication | 1.000 |
| encoding_obfuscation | 1.000 |
| hallucination_trigger | 1.000 |
| jailbreak | 1.000 |
| multilingual_evasion | 1.000 |
| ood_input | 1.000 |
| phi_exfiltration | 1.000 |
| prompt_injection | 1.000 |
| tool_misuse | 1.000 |

## Pass-rate by severity

| Severity | Pass rate |
|---|---|
| critical | 1.000 |
| high | 1.000 |
| moderate | 1.000 |
| low | 1.000 |

## Failures (first 25)

| ID | Cat | Sev | Observed | Expected | Forbidden hits |
|---|---|---|---|---|---|

## Rationale

v2 red-team corpus 'redteam-v2-2026-04' against target 'detect_phi': 110 case(s), overall pass-rate 1.000. All cases passed.

## References

- Greshake K et al. Not what you've signed up for. arXiv:2302.12173 (2023).
- OWASP LLM Top 10 -- LLM01 + LLM02 (2024).
- HHS HIPAA Privacy Rule (45 CFR §§ 164.500-534).
