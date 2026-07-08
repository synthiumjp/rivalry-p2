
## Cat1 source deviation (TriviaQA-only)

The registration names TriviaQA (rc.nocontext), Natural Questions, and BioASQ as
Cat1 factoid sources. The constructed benchmark draws Cat1 from TriviaQA
rc.nocontext only. Consequently the biomedical domain that BioASQ would have
contributed is effectively unrepresented: an approximate keyword categorisation
of the 800-prompt development set places under 2% of items in a biomedical
category, and manual inspection shows even that fraction is inflated by
incidental keyword matches (e.g. "born" matching a geography item), so true
biomedical coverage is lower still. This does not affect the confirmatory
hypotheses, which are domain-agnostic and evaluated on per-prompt hallucination
rate rather than any per-domain contrast; no confirmatory claim is conditioned
on domain balance. The narrowed source is recorded as a generalisation
limitation: results are established on general-knowledge factoids and are not
claimed to extend to biomedical factuality, which the H-Neuron literature treats
as a partly distinct regime. A secondary benefit is that holding the domain
distribution flat and narrow removes domain composition as a confound in the
cross-family L* comparison. The categorisation is heuristic (TriviaQA carries no
domain labels) and is reported only to bound biomedical representation, not a
precise domain histogram.
