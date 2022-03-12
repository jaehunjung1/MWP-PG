# CSE 517 Final Project

This is the repository for CSE 517 final project by Jaehun Jung, Chanwoo Kim, Zhitao Yu.

## Dependencies
You can install the requirements by `pip install -r requirements.txt`. Also, download required files for Stanza library with following lines of code:
```python
import stanza
stanza.download("en")
```

## Data & Preprocessing
The required datasets, [ASDIV-a](https://aclanthology.org/2020.acl-main.92/) and [MAWPS](https://aclanthology.org/N16-1136.pdf) are already included in `code/$model_name/data`. No further preprocessing is needed, as it's already done into a tsv format.

## Training
To train each model for each dataset, run the following command:
### Graph2Tree, GTS
```shell
cd code/{Graph2Tree, GTS}
python -m src.main -mode train -dataset ${mawps_fold0, cv_asdiv-a} -embedding_size $EMBEDDING_SIZE -hidden_size $HIDDEN_SIZE -depth 2 -lr $LR -emb_lr $EMB_LR -epochs $EPOCHS -full_cv -run_name $RUN_NAME -save_model -gpu $GPU_ID
```

Please refer to the paper for the best set of hyperparameters.

## Evaluation
Evaluation can be done with the following command:
### Graph2Tree, GTS
```shell
python -m src.main -mode test -dataset ${mawps_fold0, cv_asdiv-a} -embedding_size $EMBEDDING_SIZE -hidden_size $HIDDEN_SIZE -depth 2 -epochs $EPOCHS -full_cv -run_name $RUN_NAME -save_model -gpu $GPU_ID
```

This will create `outputs/$RUN_NAME/outputs_test.txt` in each model directory with the generated adversarial examples.

## Train & Evaluation with BART (additional experiment)
We evaluate how competitive the pretrained LM based Seq2Seq formulation of math word problems is, compared to task-specific models Graph2Tree and GTS.

The training code is as following:
```shell
cd code/BART
python main.py --device_id=$GPU_ID --dataset=${asdiv-a, mawps}
```
Note that in case of additional experiment with BART, the best set of hyperparameters are configured to be the default parameters in `main.py`. To evaluate, run the following command:

```shell
python main.py --device_id=$GPU_ID --dataset=${asdiv-a, mawps} --ckpt=$CHECKPOINT_FILENAME
```






