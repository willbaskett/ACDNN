# ACDNN
Adaptive Complexity Deep Neural Networks: A specialized CNN architecture designed specifically to maximize performance on small biomedical datasets. Intended for small 1D, 2D, and 3D datasets.

[Published as "Addressing Small Sample Size and Improving Interpretability in Biomedical Applications with Adaptive Complexity Deep Neural Networks" in the 2025 IEEE 13th International Conference on Healthcare Informatics (ICHI)](https://ieeexplore.ieee.org/abstract/document/11081529)

## ARCHITECTURE
ACDNNs consist of hierarchical sub-networks which each make separate predictions. Simple sub-networks struggle to fit the data but do not overfit. Complex sub-networks easily fit the data but may also overfit, degrading performance on unseen data. This allows ACDNNs to learn to solve problems at many different levels of model complexity simultaneously. The amount of model complexity which generalizes best can then be identified at evaluation time, rather than trying to guess it before training begins.
![ACDNN Architecture](resources/figure1.png)

## ARCHITECTURE AS A CNN
For use in CNNs, the ACDNN connective restriction is modified to work in the channel dimension.
![ACDNN-CNN Architecture](resources/figure2.png)

## ACDNN Training
The image below shows shows the training and testing loss of a trained ACDNN. Training loss (left) quickly decreases as sub-networks (paths) become more complex. Testing loss (center) however decreases to a point and then rapidly increases again as more complexity is allowed. Best performance is obtained by using the optimal amount of model complexity.
![ACDNN Training](resources/figure3.png)

## USAGE
See [the demo notebook](ACDNN-CNN-3D-demo.ipynb) for an example of usage.
