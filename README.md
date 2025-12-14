# Project 1: Zero-Shot Question Answering

## Overview

The aim of this project is to get familiar with Language Models for Zero-Shot Question Answering and possible pitfalls when it comes to measuring LLM performance on common benchmarks.

Zero-shot question answering is a task where a model is given a question and a context, and the model is expected to predict the answer without any training on the context or the question. The model is expected to generalize to unseen context and questions. From a practical perspective, it is a situation where we want to use a model for our task without any fine-tuning.

The project is divided into two parts:

- **Encoder Models**: Using predefined HF / transformers classes to solve this task.
- **Decoder Models**: Adapting a decoder model to solve this task and evaluating modern LLMs on this benchmark.

---

## Part 0: Setup

```bash
%pip install datasets
%pip install 'transformers[torch]'
```

## Part 1: Dataset

We use the **MMLU dataset**. Each question includes:
- 4 answer options
- Correct answer
- Subject

⸻

### Part 2: Encoder Models

**How it works**

Out-of-the-box transformer pipelines use ModelForSequenceClassification. The base model is **ModernBERT**, fine-tuned for QA. Labels:
```json
"id2label": {
    "0": "entailment",
    "1": "not_entailment"
}
```
The model classifies each option as entailment or not. The option with the highest entailment score is selected.

**Tasks**
	1.	Implement a naive function to evaluate the dataset row by row.
	2.	Implement a **batched** version for efficiency.
	3.	Compare naive vs batched results.
	4.	Profile the batched version and select the best batch size.
	5.	Calculate metrics like accuracy and per-subject analysis.

Example evaluation code:
```python
def evaluate_accuracy(result, dataset):
    correct = 0
    for i in range(len(dataset)):
        correct_answer = ["a", "b", "c", "d"].index(dataset[i]["answer"].lower())
        if result["top_inds"][i][0] == correct_answer:
            correct += 1
    accuracy = correct / len(dataset)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy
```
Profiling:
- Naive implementation: 17.4s, ~1.84 examples/sec
- Batched implementation: 16.6s, ~1.93 examples/sec
- Batch size selected: 8

## Part3: Decoder Models
Decoder models predict text autoregressively. Challenges:
- Model may not output expected format.
- Probabilities for each answer are not directly available.

**Perplexity-Based Evaluation**
Perplexity measures the likelihood of a sequence:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
```
**Prompt Template**
```python
from textwrap import dedent

HELM_PROMPT_TEMPLATE = dedent("""
The following are multiple choice questions (with answers) about {subject}:

{question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Answer:
""")
```
**Sequential KV Cache Evaluation**
```python
def compute_unnormalised_log_prob_sequentially(model, tokenizer, prompt, completions, correct):
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**prompt_inputs, use_cache=True)
        prompt_kv_cache = outputs.past_key_values

    log_probs_list = []
    for completion in completions:
        completion_inputs = tokenizer(completion, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=completion_inputs.input_ids, past_key_values=prompt_kv_cache, use_cache=True)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        input_ids = completion_inputs.input_ids[0]
        seq_log_prob = sum([log_probs[0, i, token_id] for i, token_id in enumerate(input_ids)])
        log_probs_list.append(seq_log_prob)

    correct_index = ord(correct.upper()) - ord("A")
    is_correct = np.argmax(log_probs_list) == correct_index
    return np.array(log_probs_list), is_correct
```
**Vectorized KV Cache Evaluation**
```python
def compute_unnormalised_log_prob_vectorized(model, tokenizer, prompt, completions, correct):
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**prompt_inputs, use_cache=True)
        prompt_kv_cache = outputs.past_key_values

    tokenizer.pad_token = tokenizer.eos_token
    completion_inputs = tokenizer(completions, return_tensors="pt", padding=True).to(model.device)
    input_ids = completion_inputs.input_ids
    attention_mask = completion_inputs.attention_mask
    batch_size = input_ids.size(0)

    batched_prompt_kv_cache = tuple([
        (k.expand(batch_size, *k.shape[1:]).contiguous(), v.expand(batch_size, *v.shape[1:]).contiguous())
        for (k, v) in prompt_kv_cache
    ])

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=batched_prompt_kv_cache, use_cache=True)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    log_probs_list = []
    for i in range(batch_size):
        seq_log_prob = 0.0
        for t in range(input_ids.shape[1]):
            token_id = input_ids[i, t]
            if token_id == tokenizer.pad_token_id:
                continue
            seq_log_prob += log_probs[i, t, token_id]
        log_probs_list.append(seq_log_prob)

    correct_index = ord(correct.upper()) - ord("A")
    is_correct = np.argmax(log_probs_list) == correct_index
    return np.array(log_probs_list), is_correct
```
