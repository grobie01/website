---
title: Synethic Languages and LLMs
layout: blogpost
tags: post
---
# How LLMs Understand Synthetic Languages

I began learning Finnish in 2014, a time when Google Translate was (despite the warnings from my high school Spanish teacher) relatively good for the most widely spoken languages, and rather poor for the less common ones (like Finnish). Computer translators at that point were reliant on heuristic methods , which were limited both by the complexity of the language, and the amount of effort put into the specific heuristic for a given language. Finnish, for computer translators was an unfortunate combination of high complexity and low investment interest. 

To understand the complexity of Finnish, you must first understand where it sits on a linguistic spectrum with “[analytic](https://en.wikipedia.org/wiki/Analytic_language)” on one end and “[synthetic](https://en.wikipedia.org/wiki/Synthetic_language)” on the other. In analytic languages, like English and Chinese, individual concepts map tightly to single words, and you rearrange these words in order to form complex ideas. In a more synthetic language, concepts combine to form entirely new words. In a highly synthetic language like Finnish, very complex ideas can be expressed with a single word.

Language models like BERT and GPT use neural networks to predict the next word or token in a sequence. When talking about English models we usually consider “word” and “token” to be basically interchangeable. These models are trained on massive amounts of text data to learn token patterns within a language. However, some languages pose unique challenges for language model training due to their complex morphological systems that greatly increase the total number of unique words that appear in training data.

Morphology refers to the internal structure of words. Languages with “productive morphology”, like Finnish, Turkish, and Inuktitut, have complex processes to build words from basic roots and affixes (prefixes and suffixes). This allows creating many distinct words from a single root. For example, from the Finnish root "juo" (drink), over 400 derived forms can be generated, like "juonut" (have drunk) and "juotavaa" (something that can be drunk). There is a famous word in Finnish, “juoksentelisinkohan”, which means “I wonder if I should run around without purpose.” Although this is a somewhat ridiculous example, it shows how just using individual words as tokens in the training data of a synthetic language might cause issues.

Texts in these morphologically complex languages contain many more unique word types than equivalent texts in morphologically simpler languages like English. [Research](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00365/98237/Morphology-Matters-A-Multilingual-Language) has quantified this difference using type-token ratio (TTR), which measures the number of unique words (types) divided by the total words (tokens). Finnish Bible translations have much higher TTR than English Bibles.

More word types make the language modeling task harder - the model has to learn to predict a much larger vocabulary. Perplexity, a common evaluation metric based on how surprised the model is by each next word, is higher for morphologically complex languages when standard training approaches are used, i.e. a naive approach that considers each unique word as a token. 

In order to create better tokens for synthetic languages, more specialized approaches are needed. One prevalent technique is [byte-pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE), which splits words into smaller subword units based on frequency. Frequent substrings become BPE tokens. However, BPE relies only on statistics, not linguistic knowledge. Models trained on BPE-segmented languages still show strong correlations between perplexity and increased complexity of typological features related to morphology. Even when encoded on specific languages, BPE does not fully mitigate the morphological complexity. But in the case of general models, there is an additional challenge where the byte-pair encodings are created based on a small representative sample of all of the training data, which means the BPE process will see very few tokens in less common languages, leading to smaller token size. We can see this by comparing an English and Finnish sentence with the same meaning. The English sentence is seven words long, and contains eight tokens, whereas the same sentence in Finnish is only three words long, but contains 22 tokens (most of which are one or two letters). 

![English.png](/assets/Finnish/English.png)

![Finnish.png](/assets/Finnish/Finnish.png)

([llama-tokenizer](https://belladoreai.github.io/llama-tokenizer-js/example-demo/build/))

More recent work explores linguistically informed segmentation methods. [Morfessor](https://morfessor.readthedocs.io/en/latest/) uses algorithms to induce lexical units similar to morphemes. [Finite-state transducers](https://en.wikipedia.org/wiki/Finite-state_transducer) use expert-generated linguistic rules to split words into constituent morphemes. These produce segments closer to true morphological units.

Experiments show Morfessor helps for most languages compared to BPE, and finite-state transducers perform even better when available. The more the segmentation accounts for morphology, the less impact morphology has on the trained model. Language modeling becomes less dependent on the typological properties.

We can see the successes of these methods in GPT-4. The model is an extremely effective translator of Finnish, something I certainly wish I had access to when I was studying the language eight years ago. Its boundaries are likely reflective of its training data. It stumbles on providing example sentences for particular grammatical structures of a given word. For example, if we ask for an example sentence with the word trombone (in Finnish, “pasuuna”) in the essive case (a declension meaning “in the manner of”), we get some nonsensical responses. The word in question “pasuunana” should itself not be totally illogical, but is perhaps so rare in the training set that it becomes difficult to extrapolate its normal context. 

The extensive word formation in morphologically rich languages creates challenges for training performant language models. For models like GPT which are trained using multi-language data sets, BPE is used, but primarily reflects the languages which make up the majority of its training data. This is part of why we see higher performance with English and languages with similar morphology — the languages can share similar encodings. As we get to more morphologically distinct languages, especially synthetic languages like Finnish, these byte-pair encodings become less useful, as we approach character-by-character predictions. Using richer systems of tokenization and segmentation based on the different types of linguistic complexity is an effective strategy to handle morphology and potentially improve language modeling across diverse languages.