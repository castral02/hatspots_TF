# Trained Model

HatSpot is a trained model that was trained on data from [DelRosso & Suzuki et al., 2024](https://www.biorxiv.org/content/10.1101/2024.08.19.608698v1). In order for the model to converge, Kd values were transformed into ln as logarithmic values, monotonically scaled with larger values. HatSpot has two major input nodes: one for the domain and the other for the transcription factor. Each head consists of a linear layer followed by batch normalization and ReLU activation. After concatenation, the combined representation passes through another hidden layer (linear → batch normalization → ReLU → dropout) before the final output layer, which uses no activation function to directly output ln(Kd) values.

## Workflow Outline
![Workflow]()

## Training Metrics

![Training](../images/training_metrics.png)
