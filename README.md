# Open-Vocabulary Sound Event Localization and Detection (SELD)

This codebase is the source code for "Open-Vocabulary Sound Event Localization and Detection with Joint Learning of CLAP Embedding and Activity-Coupled Cartesian DOA Vector."

## Abstract

We aim for an open-vocabulary sound event localization and detection (SELD) system that detects and localizes sound events in any category described by prompt texts. An open-vocabulary SELD system can be applied to various SELD tasks by changing prompt texts. A simple approach is to combine a language-audio model, such as a contrastive language-audio pretraining (CLAP) model, and a direction-of-arrival (DOA) estimation model. However, this combining approach cannot tackle overlapping sound events because the language-audio models output only one audio embedding, even for an audio input containing multiple events. Also, such a naive combination of two models can be sub-optimal regarding joint localization and detection. In this study, we present an embed-ACCDOA model, which jointly learns to output an embedding and the corresponding activity-coupled Cartesian DOA (ACCDOA) vector of each event in a track-wise manner, thereby tackling overlapping events. Each event’s embedding is trained to align with its corresponding audio and text embeddings inferred by a pretrained language-audio model, distilling its knowledge into the embed-ACCDOA model. We evaluate the proposed embed-ACCDOA model using an original synthetic dataset and two external SELD datasets. The knowledge distillation using both audio and text embeddings performs better than distillation using only one of the embeddings. The embed-ACCDOA model outperforms the naive combination. Moreover, it performs better than the official baseline system trained on the fully annotated training data of the target categories.

## Code

Under preparation.
