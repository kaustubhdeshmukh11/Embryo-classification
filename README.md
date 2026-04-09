Embryo Development Stage Classification
This repository contains a deep learning pipeline for the automated classification of human embryo developmental stages from time-lapse video frames. The project implements multiple convolutional neural network (CNN) architectures and a custom biologically-aware loss function to handle the sequential nature of embryo development.


Custom Loss FunctionThe project uses an ordinal distance penalty:

$$L_{total} = L_{CE} + \alpha \cdot \text{mean}(\sum p_k \cdot |pos_k - pos_{gt}|)$$

This ensures that if the model misclassifies a stage, it is incentivized to predict a biologically adjacent stage.




Data Pipeline-

Training Set: 210,600 frames (492 videos)
Validation Set: 44,436 frames (106 videos)
Test Set: 42,392 frames (106 videos)
