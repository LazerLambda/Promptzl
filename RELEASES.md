# Releases

# 1.0.0 Promptzl Stable Version

Stable version of promptzl. Turn LLMs into classifers!

### Breaking Changes
 - Class `promptzl.prompt.FVP` renamed to `promptzl.prompt.FFnVbzPair`
 - Argument `tok_args` renamed to `tokenizer_args` in `promptzl.modules.CausalLM4Classification` and `promptzl.modules.MaskedLM4Classification`

### Bugfixes
 - Throw error if empty labels in verbalizer are passed.
 - Add error if prompt is too long upon initialization

## 0.9.3

### Add Option to Return (Grouped) Logits
 `classify` method allows to pass the parameter `return_logits`. This option will set the grouped
 logits to the `LLM4ClassificationOutput` class. In case of multiple words for a label (e.g., `Vbz([['good', 'great'], ['bad']])`),
 the logits are averaged before return (i.e., The logits for `['good', 'great'],` will be averaged).

### Bugfixes
 - Fix verbalizer not updated after calling `set_prompt`.
 - Fix bug when using Dict[str, List[str]] as a verbalizer with the `Vbz` class.
