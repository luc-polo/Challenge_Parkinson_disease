# Transformer details

## Parameters optimization

### Seeds

Seeds are used to assess stability, not optimized to select the luckiest model. The same five patient-separated validation splits are repeated with encoder seeds **42, 43, and 44**, giving **15 runs**. Each seed controls initialization, training randomness, and the internal holdout split. Average validation MSE and its variability summarize performance. Final training currently uses the fixed seed **42**.

### Number of epochs

Epoch selection happens **inside each validation split, separately for each seed**:

1. Reserve **20% of that split's training patients** as an internal holdout. Fit the encoder's preprocessing and scaling on the remaining patients only.
2. Train for at most **100 epochs**, stopping after **10 consecutive epochs without improvement** in internal-holdout MSE. Retain the epoch count achieving the lowest MSE.
3. Refit preprocessing and scaling on **all training patients of that split**, including the internal holdout. Train a fresh encoder for exactly the selected number of epochs.
4. Freeze the encoder, append its embeddings, train the unchanged HistGB, and evaluate on the separate validation patients.

Validation patients never select the epoch count. Final training repeats this internal selection process on the full training dataset, then refits the encoder and HistGB; test patients remain excluded.
