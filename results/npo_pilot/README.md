# NPO pilot (exploratory, not pre-registered)

SafeTune NPO with LoRA (rank 16) on Tiny Aya Global, seed 0, through the official tokenizer, at 17 settings, scored at fp16 and after Q4_0 and NF4 round trips. We ran it to see whether a gradient-based method relapses in 4-bit files where RMU does not. It did not give a usable comparison: at lower learning rates it did not forget, at higher ones it lost general knowledge as fast as hazardous knowledge, and its 4-bit copies scored like fp16 throughout. `results.json` has every row.
