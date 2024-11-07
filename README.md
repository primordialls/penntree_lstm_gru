# Language Modeling with LSTMs and GRUs on the Penn Tree Bank Dataset

This README provides step-by-step instructions on how to train and evaluate language models using LSTM and GRU architectures on the Penn Tree Bank (PTB) dataset. The experiments cover four settings:

1. LSTM without dropout
2. LSTM with dropout
3. GRU without dropout
4. GRU with dropout

## Prerequisites

- **Python 3.6 or higher**
- **PyTorch 1.7 or higher**

### Additional Python Libraries:
- `matplotlib`
- `zipfile`

## Data Preparation

The script will automatically extract and preprocess the PTB dataset from the `ptb_data.zip` file. It will:
- Read each line of the dataset files.
- Tokenize the text and add `<eos>` tokens at the end of each line.
- Build a vocabulary from the training data.
- Convert tokens to indices based on the vocabulary.

No additional steps are required for data preparation.

## Training the Models

### Experiment 1: LSTM without Dropout
**Description**: Trains an LSTM-based language model without using dropout.

**Model Configuration**:
- Model Type: LSTM
- Dropout Probability: 0.0
- Embedding Dimension: 200
- Hidden Dimension: 200
- Number of Layers: 2

**Training Details**:
- Batch Size: 20
- Sequence Length: 20
- Learning Rate: Starts at 4.0
- Epochs: 25

**Instructions**:
- The script automatically runs this experiment first.
- Training logs and perplexity values will be displayed in the console.
- The best model weights are saved as `best_model.pt` after each epoch if validation loss improves.

### Experiment 2: LSTM with Dropout
**Description**: Trains an LSTM-based language model with dropout applied.

**Model Configuration**:
- Model Type: LSTM
- Dropout Probability: 0.5
- Embedding Dimension: 200
- Hidden Dimension: 200
- Number of Layers: 2

**Training Details**: Same as Experiment 1.

**Instructions**:
- This experiment runs automatically after Experiment 1.
- Training logs and perplexity values will be displayed in the console.
- The best model weights are saved as `best_model.pt`.

### Experiment 3: GRU without Dropout
**Description**: Trains a GRU-based language model without using dropout.

**Model Configuration**:
- Model Type: GRU
- Dropout Probability: 0.0
- Embedding Dimension: 200
- Hidden Dimension: 200
- Number of Layers: 2

**Training Details**: Same as Experiment 1.

**Instructions**:
- This experiment runs automatically after Experiment 2.
- Training logs and perplexity values will be displayed in the console.
- The best model weights are saved as `best_model.pt`.

### Experiment 4: GRU with Dropout
**Description**: Trains a GRU-based language model with dropout applied.

**Model Configuration**:
- Model Type: GRU
- Dropout Probability: 0.5
- Embedding Dimension: 200
- Hidden Dimension: 200
- Number of Layers: 2

**Training Details**: Same as Experiment 1.

**Instructions**:
- This experiment runs automatically after Experiment 3.
- Training logs and perplexity values will be displayed in the console.
- The best model weights are saved as `best_model.pt`.

**Note**: For each experiment, the model's configuration and training parameters are printed at the beginning. The script uses the same filename `best_model.pt` to save the best model in each experiment. If you wish to keep the models from each experiment separately, you should modify the script to save them with different filenames (e.g., `best_model_lstm_no_dropout.pt`).

## Evaluating the Models

### Loading Saved Weights
To evaluate a trained model using the saved weights:

1. **Ensure the model weights are saved**.
   - By default, the script saves the best model weights during training as `best_model.pt`.

2. **Modify the Script to Load the Correct Weights**.
   - If you wish to evaluate a specific model, you can modify the script to load the appropriate weights file before evaluation.
   ```python
   # Load the best saved model.
   model.load_state_dict(torch.load('best_model_lstm_no_dropout.pt'))
3. **Adjust Model Configuration if Necessary**.
   - Ensure that the model architecture matches the saved weights' configuration.

### Running Evaluation
To evaluate the model on the test set:

1. **Modify the Script to Skip Training (Optional)**.
   - By default, the script saves the best model weights during training as `best_model.pt`.

2. **Ensure the Model is Initialized Correctly**.
   - If you wish to evaluate a specific model, you can modify the script to load the appropriate weights file before evaluation.
   ```python
   # Load the best saved model.
   model.load_state_dict(torch.load('best_model_lstm_no_dropout.pt'))
3. **Load the Saved Weights**.
   - Ensure that the model architecture matches the saved weights' configuration.
4. **Run the Evaluation Function**.
   - The evaluation on the test set is performed using the `evaluate` function:
   ```python
   test_loss = evaluate(model, test_data, criterion, seq_len)
   test_ppl = math.exp(test_loss)
   print('=' * 89)
   print('| End of evaluation | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, test_ppl))
   print('=' * 89)
5. **Execute the Script**.
   - Run the script, and it will output the test loss and perplexity.
**Example**:  Evaluating the LSTM with Dropout Model
    ```python
    # Initialize the model with the same parameters
    model = RNNModel('LSTM', vocab_size, embedding_dim=200, hidden_dim=200, num_layers=2, dropout=0.5).to(device)
    # Load the saved weights
    model.load_state_dict(torch.load('best_model_lstm_with_dropout.pt'))
    # Define the criterion
    criterion = nn.CrossEntropyLoss()
    # Evaluate on test data
    test_loss = evaluate(model, test_data, criterion, seq_len=20)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print('| End of evaluation | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, test_ppl))
    print('=' * 89)
    
## Results and Plots
For each experiment, the script will:
- Print training and validation loss and perplexity after each epoch.
- Save the best model weights based on validation loss.
- Plot training and validation perplexity over epochs.
  - The plots are displayed using matplotlib.
  - The plots will show both training and validation perplexity curves for comparison. The plots are displayed automatically during script execution.
