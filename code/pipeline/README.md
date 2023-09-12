## We add a pipeline!

Input aligned paragraph pairs, then the  pipeline.sh will run everything for you!

Please format the input as [raw_input.json](https://github.com/chaojiang06/arXivEdits/blob/main/code/pipeline/raw_input.json), then run the following command in your terminal:

```sh
bash pipeline.sh
```

### The code will run:
1. Sentence alignment, output is <code>step1_sentence_alignment.json</code>.
2. Edits extraction, output is <code>step2_edit_extraction.json</code>.
3. Intention classification.
4. Merge the intention into the json file, output is <code>step4_edit_with_intention.json</code>.

### Download all the models:
By default, the code should download all checkpoints, including arxiv-bert model and arxiv-crf model. If you meet any issue in the downloading, you can use the following dropbox link:

```sh
https://www.dropbox.com/scl/fo/bva6w1k3v4c006afv55k8/h?rlkey=ev5xyazqjkeylbr9alf6md442&dl=0
```
Please put the `checkpoint` folder next to the `code` and `data` folder. The folder structure should look like this:

![image](https://github.com/chaojiang06/arXivEdits/assets/24948319/9f145e81-c23e-4805-9592-3f1bcdd39f05)


### Something you can adjust:
In the sentence alignment step, we currently merge the prediction from `simple-to-complex` and `complex-to-simple` directions by union. This method will yield high recall (coverage). If you want high precision, please adjust the following line in <code>pipeline.sh</code>. Pay attention to `--direction bi-direction-intersection`

```sh
python ../aligner/CRF_aligner.py --input ${RAW_INPUT_FILE} --output ${STEP1} --direction bi-direction-intersection
```

### Notes:
1. Please try to use GPU to speed up the computation.
2. In our paper, we use phrase alignment + rules to extract edits. Here for simplicify, we use the google [diff-match-patch package](https://github.com/google/diff-match-patch) with rules to extract edits.
