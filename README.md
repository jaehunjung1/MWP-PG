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

## Evaluation
Evaluation can be done with the following command:
### Graph2Tree, GTS
```shell
python -m src.main -mode test -dataset ${mawps_fold0, cv_asdiv-a} -embedding_size $EMBEDDING_SIZE -hidden_size $HIDDEN_SIZE -depth 2 -epochs $EPOCHS -full_cv -run_name $RUN_NAME -save_model -gpu $GPU_ID
```
This will create `outputs/$RUN_NAME/outputs_test.txt` in each model directory with the generated adversarial examples.

| **Option** | **Description** | **Default** |
|:--- | :--- | :---: |
|`mode`| Train / Test mode | 'train' |
|`dataset`| Dataset to train / test | mawps_fold0 |
|`embedding_size`| Embedding dimension | 768 |
|`hidden_size`| Hidden state dimension | 768 |
|`lr`| Learning Rate | 1e-3 |
|`emb_lr`| Embedding learning rate | 1e-5 |
|`epochs`| Number of epochs | 50 |
|`gpu`| GPU ID | 0 |

## Reproduced results
### MWP-GP with MAWPS, ASDiv-A
|  **Models** | **MAWPS-orig** | **MAWPS-QR** | **MAWPS-SP**| **ASDiv-A-orig** | **ASDiv-A-QR** | **ASDiv-A-SP** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GTS | 82.8 (82.6) | 33.2 (32.3) | 23.1 (22.7) | 71.5 (71.4) | 29.9 (30.5) | 22.7 (21.2) |
| Graph2Tree | 83.2 (83.7) | 35.8 (35.6) | 24.3 (25.5) | 76.8 (77.4) | 34.1 (33.5) | 23.4 (23.8) |

### MWP-GP with augmented MAWPS, ASDiv-A
|  **Models** | **MAWPS-Aug-QR** | **MAWPS-Aug-SP**| **ASDiv-A-Aug-QR** | **ASDiv-A-Aug-SP** |
| :---: | :---: | :---: | :---: | :---: |
| GTS - w/o BERT | 52.1 (52.3) | 41.2 (40.7) | 49.3 (48.4) | 32.1 (31.6) |
| Graph2Tree - w/o BERT | 53.8 (54.9) | 41.8 (42.3) | 55.6 (54.8) | 34.1 (33.0) |
| GTS - w/ BERT | 62.4 (63.0) | 45.0 (43.5) | 60.5 (59.8) | 40.8 (40.0) |
| Graph2Tree - w/ BERT | 64.2 (65.6) | 45.8 (45.5) | 61.9 (62.7) | 41.7 (42.6) |

Numbers in parentheses are from the original paper.


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






