# HatSpot 

This repository contains the Python code, examples, and trained model for predicting EP300-Transcription Factor Interactions using Evolutionary Scale Modeling Pooled Embeddings and experimentally known binding metrics. 


## About
EP300 is a histone acetyltransferase (HAT) enzyme that serves as a critical transcriptional co-activator for a variety of lineage-specific transcription factors (TFs). Many TFs interact with and recruit this HAT enzyme through dedicated protein-protein interactions, enabling it to acetylate histones and increase chromatin accessibility at specific genomic loci. While small molecule inhibition of EP300 has been shown to alter many specific TF-dependent gene expression programs, it remains difficult to predict what TFs are directly EP300-dependent and which may therefore be targeted therapeutically using this approach. 

To address this challenge, we developed a deep learning model capable of prioritizing EP300-TF interactions for experimental testing called **HatSpot**.

Our approach integrates recently developed high-throughput EP300-TF binding [data](https://www.biorxiv.org/content/10.1101/2024.08.19.608698v1) in combination with Evolutionary Scale Modeling (ESM) embeddings to develop a novel scoring system called **Hat_score** that significantly enhances the prediction of EP300-TF relative to traditional structure prediction metrics. We anticipate this model will be useful to prioritize the discovery of uncharacterized regulatory interactions, providing a link between high-throughput transcriptional activation assays and EP300 that will be potentially extensible to additional druggable transcriptional co-activator families.

## HatSpot Workflow

We implemented [ESM2 650M pooled Embeddings](https://www.science.org/doi/10.1126/science.ade2574) as inputs to HatSpot. Due to memory and size, individual ESM embeddings for every single amino acid are unachievable; therefore, we routed to a pooled embedding to capture not only residue-level description but also overall tile/TF description. 


HatSpot has two major input nodes: one for the domain and the other for the transcription factor. Each head consists of a linear layer followed by batch normalization and ReLU activation. After concatenation, the combined representation passes through another hidden layer (linear → batch normalization → ReLU → dropout) before the final output layer, which uses no activation function to directly output ln(Kd) values.

### Workflow Diagram
![Workflow]()

---

## Declaration of Generative AI Usage

This project utilized OpenAI's ChatGPT to assist in generating Python code, documentation, and explanatory content.

## References 

- DelRosso, N., Suzuki, P.H., et al. *High-throughput affinity measurements of direct interactions between activation domains and co-activators*. **BioRxiv**, (2024). [paper link](https://www.biorxiv.org/content/10.1101/2024.08.19.608698v1)
- Lin, Z., Halil, A., et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. **Science**, 379, 1123-1130, (2023). [paper link](https://www.science.org/doi/10.1126/science.ade2574)

## Biowulf Acknowledgement: 

This work utilized the computational resources of the NIH HPC [Biowulf cluster](https://hpc.nih.gov).
