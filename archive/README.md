# System Predictions and Evaluation Scores

- [conll/](#conll)
    - [Explanation on the Directory Tree](#explanation-on-the-directory-tree)
- [scores/](#scores)


## [conll/](conll/)
Contains CoNLL files with gold annotations and system predictions from all reported experiments. The [unzipped](https://www.7-zip.org/) files take up roughly 830MB.

Each file (with the exception of lemma baseline files) has the following columnar format:
1. topic identifier
2. subtopic identifier
3. document identifier
4. sentence index
5. token index (1-based, resets every document)
6. token
7. `True`
8. coreference chain

For legal reasons concerning the FCC-T corpus, every token but the first in each sentence is replaced with "___" (see [conll/strip_tokens.py](conll/strip_tokens.py)).

### Explanation on the Directory Tree
```
conll/
├── key                                # key files with gold annotations
│   ├── ecbp                           
│   ├── fcct                           
│   └── gvc                            
├── sys_cdcr                           # system predictions from CDCR experiments
│   ├── gg                             # CoNLL files for scenario S_gg
│   │   ├── ecbp                       # training corpus is ECB+
│   │   │   └── events                 
│   │   │       ├── ecbp_on_ecbp       # model trained on ECB+, tested on ECB+
│   │   │       │   ├── closure        # test split where documents were pre-clustered by the transitive closure of gold-annotated coreference clusters
│   │   │       │   │   ├── 0          # CoNLL file from model trained/optimized with random seed 0
│   │   │       │   │   ├── 1          
│   │   │       │   │   └── 2          
│   │   │       │   └── predicted      # test split where documents were preclustered by textual content as done in Barhom et al. 2019
│   │   │       └── ecbp_on_fcct       # model trained on ECB+, tested on FCC-T
│   │   │           └── predicted      # for FCC-T, transitive closure and predicted clustering are identical
│   │   └── ...                        
│   ├── gg_aug                         # CoNLL files for scenario S_gg with silver-augmented training data
│   │   ├── gg_aug_abc-ecbp            # ECB+ training data augmented with ABC News
│   │   ├── gg_aug_abc-fcct            # FCC-T training data augmented with ABC News
│   │   └── ...                        
│   ├── sg                             # CoNLL files for scenario S_sg
│   │   ├── lemma-delta                # lemma-delta baseline
│   │   ├── sg_abc_25k                 # training corpus is ABC News
│   │   └── sg_bbc_25k                 # training corpus is BBC News
│   └── ss                             # CoNLL files for scenario S_ss
│       ├── lemma                      # lemma baseline
│       ├── ss_abc_25k                 
│       └── ss_bbc_25k                 
└── sys_mentiondetection               # system predictions from event mention detection experiments
    └── ...                            
```

## [scores/](scores/)
Our full experiment results for every experiment, scenario, training/dev corpus, preclustering choice, random seed, measure and metric.

Jump start with pandas:
```python3
>>> import pandas as pd

>>> pd.read_csv("sys_cdcr_all_scores_unaggregated.csv",
                header=list(range(2)),
                index_col=list(range(6)))

measure                                                     mentions       ...                conll
metric                                                            F1    P  ...         R         F1
scenario  experiment  dev-corpus test-corpus topics    seed                ...                     
gg        ecbp        ecbp       gvc         closure   0         1.0  1.0  ...  0.648174  59.593900
                                                       1         1.0  1.0  ...  0.517531  63.448063
                                                       2         1.0  1.0  ...  0.571457  64.668658
                                             predicted 0         1.0  1.0  ...  0.605251  54.988492
                                                       1         1.0  1.0  ...  0.467913  58.908891
...                                                              ...  ...  ...       ...        ...
```