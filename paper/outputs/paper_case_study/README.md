# Case study of the paper (operando NMC622, Tallman et al. 2021)

Produced by `python paper/case_study.py --rebuild --target 25` on 2026-09-24 with the OpenRouter free tier.

- corpus: `corpora/ni_k_nmc/corpus_manifest.json` (agent queries, every screening score/reason/model, exclusions, documents)
- `facts.txt` - numbers handed to the LLM (no reference labels)
- `prompt.txt` - exact prompt (facts + 15 retrieved passages + instructions)
- `paragraph.txt` - verbatim output of GLM-5.2 (z-ai/glm-5.2:free)
- `audit.json` - claim audit by Nemotron 3 Ultra (nvidia/nemotron-3-ultra-550b-a55b:free)
- `run.json` - models, retrieval scores, deterministic numeric check, audit summary
- `hits_metadata.json` - retrieved passages without their text (see prompt.txt for the excerpts)

The paper whose spectra are analysed (doi:10.1021/acs.jpcc.0c08095) is excluded from the corpus by DOI and title.
