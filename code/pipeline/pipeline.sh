
RAW_INPUT_FILE="raw_input.json"
STEP1="step1_sentence_alignment.json"
STEP2="step2_edit_extraction.json"
STEP4="step4_edit_with_intention.json"

# step 0, download all models.
git lfs install
git clone https://huggingface.co/chaojiang06/arxiv-sentence-alignment ../../checkpoints/arxiv-sentence-alignment

# step 1, load the raw input file and perform sentence alignment.
python ../aligner/CRF_aligner.py --input ${RAW_INPUT_FILE} --output ${STEP1}

# step 2, extract edits from the aligned sentence pairs.
python ../edits/extract_edits.py --input ${STEP1} --output ${STEP2}

# step 3, run the intention classifier on the edits.

python -u ../intention/run_translation_arxiv.py \
  --model_name_or_path chaojiang06/arXivEdits-intention-classifier-T5-large-fine-grained \
  --do_eval \
  --do_predict \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --num_train_epochs 10 \
  --source_prefix "intent classification: " \
  --train_file ../../data/edits/train.json\
  --validation_file  ../../data/edits/dev.json\
  --test_file ${STEP2}\
  --output_dir ../intention/tmp \
  --per_device_train_batch_size=12 \
  --per_device_eval_batch_size=12 \
  --metric_for_best_model accuracy \
  --overwrite_output_dir \
  --overwrite_cache \
  --predict_with_generate



# step 4, merge the predicted intention with the edits, output the final file.

python -u merge.py --input ${STEP2} --intention ../intention/tmp/generated_predictions.txt --output ${STEP4}