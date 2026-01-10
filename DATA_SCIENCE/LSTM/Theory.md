<h3 align="center">Long Short-Term Memory (LSTM)</h3>

  <p align="center">
    Explanation of structure and logic behind memory gates, examples of usage
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>


## Table of contents

- [Explanation](#explanation)
- [Examples of use](#examples-of-use)
- [Code implementation](#code-implementation)


## Explanation
**Recurent Neural Networks**
Simply put, RNNs are networks with loops in them, allowing **information to persist**. After unrolling RNN (image below) they can be thought as multiple copies of the same network, each passing a message to a successor. 
![](/resources/imgs/RNN.png)
The problem is, RNNs don't have any logic implementation on *which* and *how* they should remember informations passed to them in different scenarios, like where they need context from way back. They work well only if the gap between prediction and revelant information is small. 
Example: 
- The clouds are in the `sky` <- no problem predicting based on previous words only
- I used to live for 10 years in France ..... I speak fluent `French`, to predict this one we need context about France from further back, but this information wasn't retained in memory


**LSTMs solving the problem**

LSTMs are special kind of RNNs, capable of learning **long-term dependencies** in sequential data where order matters (kinda like undestanding the context of the whole sentence/paragraph rather just remembering one word before last word).

- They solve the problem of RNNs struggling with remembering long-term dependencies in data (that can often be crucial for making correct predictions) caused by **vanishing/exploding gradient problem** (when during training gradient either shrink or grow too large causing model to either think that earlier information is irrevelant or too important, no real learning is being done in that scenario) by their **use of gates** that decide how to maintain informations and pass it through cells optimaly:
    - **Forget Gate** - decides what information to remove from cell state 
    - **Input Gate** - decides what to take from short term and add to long term (cell state)
    - **Output Gate** - decides which parts of the cell state to output (push to another cell)
<br>
- LSTM cell structure:

![](/resources/imgs/LSTM_1.png)

- Gates explained:

Forget Gate     |  Input Gate 
:-------------------------:|:-------------------------:
|![](/resources/imgs/LSTM_2.png) | ![](/resources/imgs/LSTM_3.png)

Candidate Hidden state    | Output Gate 
:-------------------------:|:-------------------------:
![](/resources/imgs/LSTM_4.png) | ![](/resources/imgs/LSTM_5.png)

## Examples of use
`Sentiment Analysis` `Next word prediction` `Language translation` `Time series forecasting` `Speech recognition`

## Code implementation
`source: Geeksforgeeks`
[ its neccessary to prepare data in a way that its sequential, before training ]
> Tensorflow
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```
```python
model = Sequential()
model.add(LSTM(units=128, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
```
---
> PyTorch
```python
import torch
import torch.nn as nn
```
```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # processes sequential data
        self.fc = nn.Linear(hidden_dim, output_dim) # maps hidden state outpusts to predictions

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, hn, cn
```
