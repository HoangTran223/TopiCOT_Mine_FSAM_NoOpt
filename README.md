# This repo forked from TopMost!

## Preparing libraries
0. Python 3.10
1. Install the following libraries
    ```
    numpy 1.26.4
    torch_kmeans 0.2.0
    pytorch 2.2.0
    sentence_transformers 2.2.2
    scipy 1.10
    bertopic 0.16.0
    gensim 4.2.0
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to ./evaluations/pametto.jar
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./data/wikipedia/ as an external reference corpus.
    ```
        |- wikipedia
            |- wikipedia_bd
                |- ...
                |- wikipedia_bd.histogram
    ```

Note: step 1, 2, and 3 can be done if conda is installed and run 
    ```
    bash setupenv.sh
    ```

## Usage
To run and evaluate our model, run the following command:

```
python main.py    --dataset [20NG|YahooAnswers|IMDB|AGNews] \
                    --model OTClusterTM \
                    --num_topics 50 \
                    --num_groups [20|10|2,3,4|2,3] \
                    --dropout 0 \
                    --seed 0 \
                    --beta_temp 0.2 \
                    --epochs 500 --device cuda --lr 0.002 --lr_scheduler StepLR \
                    --batch_size 200 --lr_step_size 125 --use_pretrainWE  \
                    --weight_ECR 250 --alpha_ECR 20 \
                    --weight_DCR 40 --alpha_DCR 20 \
                    --weight_TCR 200 --alpha_TCR 20 \
                    --wandb_prj [Name of project to save on wandb] \
```

### Parameters
- 20NG: ~ 250
- YahooAnswers: ~ 40|60
- IMDB: ~100-150
- AGNews: ~100-150
- The alpha_*CRs should keep to be equal to 20

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.
