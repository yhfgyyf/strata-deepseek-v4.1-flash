# Working on strata-deepseek-v4.1-flash

This is an experimental DeepSeek V4.1 Flash source snapshot for Linux x86-64,
CUDA 13.2 and one RTX PRO 6000 Blackwell (SM120).

- Read the relevant source and documentation before editing. Preserve local model files and configuration.
- Keep the fixed prefill chunk at 8192 unless the user asks to change it.
- Preserve model, tokenizer and quantization contracts. Do not silently substitute a checkpoint or change arithmetic.
- Never expose an unauthenticated API beyond localhost.
- Check API changes with the relevant `serve/test_*.py` tests. Check packaging configuration with `python3 -m unittest tools.deepseek41.test_release_config`.
- Native builds require Linux/CUDA. Distinguish a successful build, operator accuracy, model quality and measured throughput.
- The published measurements have explicit consistency limitations in `docs/PERFORMANCE.md`. Do not remove them or present the snapshot as production validated.
- Keep weights, private prompts/outputs, credentials and raw traces outside Git.
