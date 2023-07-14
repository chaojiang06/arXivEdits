

# arXivEdits 

The data for our EMNLP 2022 paper [**arXivEdits: Understanding the Human Revision Process in Scientific Writing**](https://arxiv.org/abs/2210.15067) is provided at this repo. 

The name of each field should be self-explainable. If you have any questions, please reach me at <chaojiang06@gmail.com>.

The code for extracting plain text from latex source code was written by awesome [Sam Stevens](https://samuelstevens.me/) when he was an undergraduate student. The raw code can be found [here](https://github.com/samuelstevens/arxiv-edits).

## Update on 2023/07/14
We upload all fine-tuned T5 intention classification models to the huggingface hub. The code is in the [code](https://github.com/chaojiang06/arXivEdits/tree/main/code/intention) folder

## Update on 2023/02/22
We upload all fine-tuned [BERT checkpoints](https://huggingface.co/chaojiang06/arxiv-sentence-alignment) to the huggingface hub and provide [a sample code](https://colab.research.google.com/drive/1-6hWzTIgrEMrcervG_ANqrf1o2CugnfS?usp=sharing) to use them.

## Update on 2023/01/23

We add license information every version of all papers. For example:

        train['1608.00087']['license'] = {'1': 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/', '2': 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'}

In total, we find the following licenses:

        http://arxiv.org/licenses/assumed-1991-2003/
        http://arxiv.org/licenses/nonexclusive-distrib/1.0/
        http://creativecommons.org/licenses/by-nc-sa/4.0/
        http://creativecommons.org/licenses/by-sa/4.0/
        http://creativecommons.org/licenses/by/3.0/
        http://creativecommons.org/licenses/by/4.0/
        http://creativecommons.org/licenses/publicdomain/
        http://creativecommons.org/publicdomain/zero/1.0/

We also add the source arxiv-id for each sentence pair in the edits sub-dataset.

Thanks for the suggestions from Qian Ruan from the UKP Lab!

### Reference

If you find our paper or dataset useful, please considering cite the following paper.

```
@article{jiang-etal-2022-arXivEdits,
  title={arXivEdits: Understanding the Human Revision Process in Scientific Writing},
  author={Jiang, Chao and Xu, Wei and Stevens, Samuel},
  journal={In Proceedings of EMNLP 2022},
  year={2022}
}
```
