# Transformer-Based Relationship Inference Model for Household Object Organization by Integrating Graph Topological and Ontological Information

**This code repository is the code for the thesis "Transformer-Based Relationship Inference Model for Household Object Organization by Integrating Graph Topological and Ontological Information", this thesis uses GAT to obtain the topology information of the household item relationship graph as well as BERT to obtain the ontology attribute information of the household items, after fusing the two kinds of information, it uses the Transformer model to train an item-relationship inference model. Thus, the relationship information between items can be accurately reasoned. The dataset constructed in this paper is not open source, but provides a sample data in the file "BERT_Input.json", and our dataset is sorted according to this format.**

![](.asert/fig1.png)


## 1. INSTALLATION

First, install the dependent python environments using the following commands

```
conda env create -f environment.yml
```

## 2.USAGE

Although we did not provide the original dataset, we provide the topological eigenvectors after encoding the original dataset with the GAT, GAE, GCN, and GraphSAGE models, which are available for download at the following address:

[GAT_Embedding](https://drive.google.com/file/d/184eXiJzMicK9FfV6B8n-wVQF9CRnu_Y3/view?usp=drive_link)

[GAE_Embedding](https://drive.google.com/file/d/1HFfmTWXn_yi_2vyXVLaOkrNKrQhgolXm/view?usp=drive_link)

[GCN_Embedding](https://drive.google.com/file/d/1LNw1ILJZCrZr_AKRdzRhUWfGG48crqQG/view?usp=drive_link)

[GraphSAGE_Embedding](https://drive.google.com/file/d/1OEmNnomTIDu5CBu3FjYVDt6Aha1SpZD6/view?usp=drive_link)

[BERT_Input](https://drive.google.com/file/d/1eLesvGxcMn-0pk8y0VpUd7WCAt_m47-q/view?usp=drive_link).

Download the above json file and put it in the root directory

After downloading the above files, you can run the following commands to train the model:

```
python train.py
```