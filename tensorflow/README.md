# README

This project contains vertex pipeline built using Keras, Tensorflow and Kubeflow.  It also contains the Dockerfiles for
building the containers use as part of the pipeline

Some noteable pipeline projects include:
- The 'pipeline_deploying_to_endpoint' pipeline file will walk you through each stage of model creation from data wrangling 
to training/testing to deploying to a vertex endpoint for inference, all the while using multi GPU accelerators.
- The 'tpu-pipeline' will show you how to leverage Google's TPU accelerators for training.
- The 'resent_pipeline' shows how to use an off the shelf model instead of creating your own
- The 'simple-model-pipeline' demonstrates how to create a DAG with two different GPU configurations so you can compare
model training time and show the better more cost effect GPU configuration for you projects.