# TinyLLM

TinyLLM is a minimal educational language model implementation in C#. The console application provides a simple workflow to train a character-level model on Shakespeare text.

## Training Example

Run the console application and choose the training option. Logs will be written to `tinyllm_training.log` and small text samples are generated after every epoch using the configured temperature (default `0.7`). Metrics CSV logging is behind a feature flag and disabled by default.

Training can take a long time, especially for large datasets. The provided Shakespeare dataset contains about one million characters which results in tens of thousands of batches per epoch.
