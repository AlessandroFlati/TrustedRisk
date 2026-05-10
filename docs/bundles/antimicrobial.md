# Bundle: `antimicrobial` -- Antimicrobial Stewardship

Empiric antibiotic selection with local antibiogram + de-escalation with pathogen-driven narrowing + IDSA "5 Cs" IV-to-PO switch criteria.

## When to use

- Sepsis admission: pick empiric coverage by source + severity.
- Day-3 stewardship review: cultures back, narrow regimen?
- IV-to-PO switch decision when patient improves.

## Typical tool sequence

1. compute_empiric_antibiotic_selection(infection_source, severity, factors, antibiogram) -- initial regimen
1. compute_antibiotic_de_escalation(current_regimen, pathogen, susceptibility, clinical_factors) -- narrow + switch
1. ground_claim(claim_text) -- cite IDSA / ATS / SHEA guidelines

## Minimum inputs

- Infection source (urinary / pneumonia / SSTI / etc.)
- Severity (uncomplicated / complicated / sepsis / septic_shock)
- Patient factors (allergies, eGFR, MRSA risk, ESBL history)
- Local antibiogram (optional but high-value)
- For de-escalation: pathogen + susceptibility S/I/R per drug

## Composed output

AntibioticSelection (ranked options) + DeEscalationPlan (target regimen + duration).

## Demo scenarios

- K (complicated UTI + ESBL)

## Tools in this bundle

- `compute_empiric_antibiotic_selection`
- `compute_antibiotic_de_escalation`
- `ground_claim`

---

_Bundle id_: `antimicrobial` &middot; _Tools_: 3 &middot; _Auto-generated from `mcp_server.tools.BUNDLES`_.
