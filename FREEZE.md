# Development freeze — Paper 2 (v6.3)

Development-set confirmatory analyses frozen at commit f184001, 15 July 2026,
before opening the sealed 200-prompt hold-out (hold_out_opened=false at freeze).

Frozen hypotheses (development results, tracked artifacts):
- H1 encoding-time prediction: data/h1_encoding_dev.json — 4/4 dev peak cv-r >0.5
  (Mistral 0.501/L31, Qwen 0.608/L21, Llama 0.564/L32, Gemma 0.589/L40);
  signal present from L1, plateau onset (0.05 tol) L15/18/16/25. Peak layer per
  model is the locked hold-out read layer.
- H2 commitment concentration: data/commitment_v2_*_instruct.json (x4).
- H3 commitment ordering: data/h3_ordering_dev.json — disconfirmed 4/4, |d|<0.11,
  no consistent ordering (d = incorrect minus correct: +0.107/-0.082/-0.104/+0.004).
- H4 stage coupling: data/h4_stage_coupling_dev.json — null 4/4, Spearman rho ~0.09.
- H5 5a linear correspondence: data/hneuron_correspondence_v2_*_last.json,
  data/layer_dynamics_v2_*_instruct.json (x4) — null 4/4.

Off the hold-out: H5 5b (ill-posed as registered, Path A), H6 (demoted to
non-confirmatory). Rationale in the deviations-and-decisions record.

Producers: scripts/h1_encoding_dev.py, scripts/h3_ordering_dev.py,
scripts/h4_stage_coupling_dev.py, scripts/commitment_confirmatory.py,
scripts/hneuron_layer_correspondence.py. H1/H3/H4 results regenerated from
tracked inputs during this freeze.

Deviations-and-decisions record (length-confound single-token fix, 5b four
defects, H6 demotion, H1 plateau reporting, H3 disconfirmation) maintained
outside the repo.
