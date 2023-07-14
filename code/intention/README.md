Please use the following command to run the code
```sh
python -u -m pdb run_translation_arxiv.py \
  --model_name_or_path t5-base \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --num_train_epochs 10 \
  --source_prefix "intent classification: " \
  --train_file ../../data/edits/train.json\
  --validation_file  ../../data/edits/dev.json\
  --test_file ../../data/edits/test.json\
  --output_dir tmp \
  --per_device_train_batch_size=12 \
  --per_device_eval_batch_size=12 \
  --metric_for_best_model accuracy \
  --overwrite_output_dir \
  --overwrite_cache \
  --coarse True \
  --predict_with_generate
```
Please adjust the `--coarse True/False` to adjust training with 4 or more classes.

All checkpoints are uploaded to the huggingface hub:

[arXivEdits-intention-classifier-T5-base-fine-grained](https://huggingface.co/chaojiang06/arXivEdits-intention-classifier-T5-base-fine-grained)

[arXivEdits-intention-classifier-T5-base-coarse](https://huggingface.co/chaojiang06/arXivEdits-intention-classifier-T5-base-coarse)

[arXivEdits-intention-classifier-T5-large-fine-grained](https://huggingface.co/chaojiang06/arXivEdits-intention-classifier-T5-large-fine-grained)

[arXivEdits-intention-classifier-T5-large-coarse](https://huggingface.co/chaojiang06/arXivEdits-intention-classifier-T5-large-coarse)


Please use the following command to reproduce [Table 5 in the paper](https://arxiv.org/pdf/2210.15067.pdf#page=8).

```sh
python -u run_translation_arxiv.py \
  --model_name_or_path arXivEdits-intention-classifier-T5-large-fine-grained \
  --do_train \
  --do_eval \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --num_train_epochs 10 \
  --source_prefix "intent classification: " \
  --train_file ../../data/edits/train.json\
  --validation_file  ../../data/edits/dev.json\
  --test_file ../../data/edits/test.json\
  --output_dir tmp \
  --per_device_train_batch_size=12 \
  --per_device_eval_batch_size=12 \
  --metric_for_best_model accuracy \
  --overwrite_output_dir \
  --overwrite_cache \
  --coarse True \
  --predict_with_generate
```

Feel free to change `--model_name_or_path` and  `--coarse` to reproduce all performance in [Table 4](https://arxiv.org/pdf/2210.15067.pdf#page=8).

When using this code for inference, please note that the key is the `preprocess_function` at line 490. Also, please consider using `['[MATH]', '[EQUATION]', '[REF]', '[CITATION]']` to replace the corresponding parts in the input.

```python
def preprocess_function(examples):
    # inputs = [ex[source_lang] for ex in examples["translation"]]
    inputs = []
    sent1_tokens_list = examples['sent1_tokens']
    sent2_tokens_list = examples['sent2_tokens']
    edit_type_list = examples['edit_type']
    edit_intention_list = examples['edit_intention']
    sent1_span_token_offset_list = examples['sent1_span_token_offset']
    sent2_span_token_offset_list = examples['sent2_span_token_offset']

    for idx, i in enumerate(edit_type_list):

        sent1_tokens = sent1_tokens_list[idx]
        sent2_tokens = sent2_tokens_list[idx]
        sent1_span_token_offset = sent1_span_token_offset_list[idx]
        sent2_span_token_offset = sent2_span_token_offset_list[idx]

        if i == 'Deletion':

            sent1_tokens[sent1_span_token_offset[0]] = '[' + sent1_tokens[
                sent1_span_token_offset[0]]
            sent1_tokens.insert(sent1_span_token_offset[1], ']')
            inputs.append(" ".join(sent1_tokens + ["|"] + sent2_tokens))
        elif i == 'Insertion':
            sent2_tokens[sent2_span_token_offset[0]] = '[' + sent2_tokens[
                sent2_span_token_offset[0]]
            sent2_tokens.insert(sent2_span_token_offset[1], ']')
            inputs.append(" ".join(sent1_tokens + ["|"] + sent2_tokens))
        elif i == 'Substitute':
            sent1_tokens[sent1_span_token_offset[0]] = '[' + sent1_tokens[
                sent1_span_token_offset[0]]
            sent1_tokens.insert(sent1_span_token_offset[1], ']')

            sent2_tokens[sent2_span_token_offset[0]] = '[' + sent2_tokens[
                sent2_span_token_offset[0]]
            sent2_tokens.insert(sent2_span_token_offset[1], ']')
            inputs.append(" ".join(sent1_tokens + ["|"] + sent2_tokens))
```
