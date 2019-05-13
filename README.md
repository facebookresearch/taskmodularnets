# Task-Driven Modular Networks

![TMN Model](http://www.cs.cmu.edu/~spurushw/projects/compositional/teaser.png)

This is the code accompanying the work:  

Task-Driven Modular Networks for Zero-Shot Compositional Learning<br/>
Senthil Purushwalkam, Maximilian Nickel, Abhinav Gupta, Marc'Aurelio Ranzato<br/>
Preprint<br/>

The repository began as a fork of the [codebase](https://github.com/Tushar-N/attributes-as-operators) which accompanies the interesting related work by Nagarajan et. al:<br/>
Nagarajan, Tushar, and Kristen Grauman. "Attributes as operators: factorizing unseen attribute-object compositions." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

## Cite

If you find this repository useful in your own research, please consider citing both papers:

```
@article{purushwalkam2019taskdriven,
  title={Task-Driven Modular Networks for Zero-Shot Compositional Learning},
  author={Purushwalkam, Senthil and Nickel, Maximilian and Gupta, Abhinav and Ranzato, Marc'Aurelio},
  journal={arXiv},
  year={2019}
}

@article{nagarajan2018attrop,
  title={Attributes as Operators},
  author={Nagarajan, Tushar and Grauman, Kristen},
  journal={ECCV},
  year={2018}
}
```


## Prerequisites
The code is written and tested using Python (3.6) and PyTorch (v1.0). 

**Packages**: Install using `pip install -r utils/requirements.txt`

**Datasets and Features**: We include a script to download all the necessary data: images, features and metadata for the two datasets, pretrained SVM classifier weights and pretrained models. It must be run before training the models. We also download the splits suggested in related work and the new splits proposed in our work. 

```bash
bash utils/download_data.sh
```

**UT-Zappos Subset**: The UT-Zappos Materials subset, which we use as one of our datasets will be generated as part of the script. The subset is also available to download as raw images from [here](https://www.cs.utexas.edu/~tushar/attribute-ops/ut-zap50k-materials.zip). 

## Training a model

Models can be trained using the train script with parameters for the model (visprodNN, redwine, labelembed+, attributeop) and the various regularizers (aux, inv, comm, ant). For example, to train the LabelEmbed+ baseline on UT-Zappos:

```bash
python train_modular.py --name tmn  --model modularpretrained --compose_type nn --batch_size 256 --softmax --lr 0.001 --lrg 0.01 --num_negs 600 --embed_rank 64 --glove_init --nmods 24 --emb_dim 16 --nlayers 3 --test_batch_size 32 --adam --pair_dropout 0.0 --pair_dropout_epoch 1 --max_epochs 5
```


## Model Evaluation

The model from the last saved epoch can be tested using the following command:

```bash
python test_modular.py --name tmn  --model modularpretrained --compose_type nn --embed_rank 64 --glove_init --nmods 24 --emb_dim 16 --nlayers 3 --test_batch_size 32 --test_set test
```

Alternatively, the model with highest validation AUC (say epoch 4 in this example), can be tested
using:

```bash
python test_modular.py --name tmn  --model modularpretrained --compose_type nn --embed_rank 64 --glove_init --nmods 24 --emb_dim 16 --nlayers 3 --test_batch_size 32 --test_set test --load models/tmn/ckpt_E_4.t7
```



See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
taskmodularnets is MIT licensed, as found in the LICENSE file.
