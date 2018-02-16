+++
title = "RNNs TV Script Generation - Deep Learning"
date = 2018-02-15T23:20:56-05:00
draft = false

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []
categories = []

summary = "In this project, we'll generate our own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  we'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network we'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern)."

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
# Use `caption` to display an image caption.
#   Markdown linking is allowed, e.g. `caption = "[Image credit](http://example.org)"`.
# Set `preview` to `false` to disable the thumbnail in listings.
[header]
image = "tvscript/bart-simpson-Regularization.gif"
caption = ""
preview = true

+++

In this project, we'll generate our own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  We'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network we'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
we'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```
## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.

```python
view_sentence_range = (0, 10)
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.248091603053435
    Number of lines: 4257
    Average number of words in each line: 11.50434578341555

    The sentences 0 to 10:
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.




## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, we first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    chars = sorted(list(set(text)))
    vocab_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_vocab = dict((i, c) for i, c in enumerate(chars))

    return vocab_to_int, int_to_vocab
```

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure we don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    values = ['||Period||','||Comma||','||Quotation_Mark||','||Semicolon||','||Exclamation_mark||','||Question_mark||','||Left_Parentheses||','||Right_Parentheses||','||Dash||','||Return||']
    keys = ['.', ',', '"', ';', '!', '?', '(', ')', '--','\n']
    return (dict(zip(keys,values)))
```
## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is our first checkpoint. If we ever decide to come back to this or have to restart, we can start from here. The preprocessed data has been saved to disk.

```python
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
We'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU

```python
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0

### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following tuple `(Input, Targets, LearningRate)`

```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input = tf.placeholder(tf.int32,[None, None], name='input')
    targets = tf.placeholder(tf.int32,[None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')

    return input, targets, learning_rate

```

### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state =  tf.identity(initial_state, name='initial_state')

    return cell, initial_state
```

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.

```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1.0, 1.0))
    embedded_input = tf.nn.embedding_lookup(embeddings, input_data)

    return embedded_input
```

### Build RNN
We created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)`

```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, name="final_state")

    return outputs, final_state
```

### Build the Neural Network
Apply the functions we implemented above to:
- Apply embedding to `input_data` using our `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and our `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState)

```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    input_data = get_embed(input_data,vocab_size,rnn_size)
    outputs,final_state = build_rnn(cell,input_data)
    logits = tf.contrib.layers.fully_connected(outputs,vocab_size,activation_fn = None)
    return logits, final_state
```

### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If we can't fill the last batch with enough data, drop the last batch.

For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2], [ 7  8], [13 14]]
    # Batch of targets
    [[ 2  3], [ 8  9], [14 15]]
  ]

  # Second Batch
  [
    # Batch of Input
    [[ 3  4], [ 9 10], [15 16]]
    # Batch of targets
    [[ 4  5], [10 11], [16 17]]
  ]

  # Third Batch
  [
    # Batch of Input
    [[ 5  6], [11 12], [17 18]]
    # Batch of targets
    [[ 6  7], [12 13], [18  1]]
  ]
]
```

Notice that the last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = int(len(int_text) / (batch_size * seq_length))
    input_data = np.array(int_text[: n_batches * batch_size * seq_length])
    target_data = np.array(int_text[1: n_batches * batch_size * seq_length + 1])
    target_data[-1] = input_data[0]

    input_batches = np.split(input_data.reshape(batch_size, -1), n_batches, 1)
    target_batches = np.split(target_data.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(input_batches, target_batches)))
```

## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 300
# Sequence Length
seq_length = 16
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 33

save_dir = './save'
```

### Build the Graph
Build the graph using the neural network we implemented.


```python
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/33   train_loss = 8.821
    Epoch   1 Batch    0/33   train_loss = 5.051
    Epoch   2 Batch    0/33   train_loss = 4.414
    Epoch   3 Batch    0/33   train_loss = 4.063
    Epoch   4 Batch    0/33   train_loss = 3.731
    Epoch   5 Batch    0/33   train_loss = 3.471
    Epoch   6 Batch    0/33   train_loss = 3.155
    Epoch   7 Batch    0/33   train_loss = 2.910
    Epoch   8 Batch    0/33   train_loss = 2.686
    Epoch   9 Batch    0/33   train_loss = 2.484
    Epoch  10 Batch    0/33   train_loss = 2.289
    Epoch  11 Batch    0/33   train_loss = 2.135
    Epoch  12 Batch    0/33   train_loss = 2.004
    Epoch  13 Batch    0/33   train_loss = 1.858
    Epoch  14 Batch    0/33   train_loss = 1.738
    Epoch  15 Batch    0/33   train_loss = 1.629
    Epoch  16 Batch    0/33   train_loss = 1.502
    Epoch  17 Batch    0/33   train_loss = 1.397
    Epoch  18 Batch    0/33   train_loss = 1.300
    Epoch  19 Batch    0/33   train_loss = 1.245
    Epoch  20 Batch    0/33   train_loss = 1.158
    Epoch  21 Batch    0/33   train_loss = 1.085
    Epoch  22 Batch    0/33   train_loss = 1.028
    Epoch  23 Batch    0/33   train_loss = 0.967
    Epoch  24 Batch    0/33   train_loss = 0.899
    Epoch  25 Batch    0/33   train_loss = 0.838
    Epoch  26 Batch    0/33   train_loss = 0.804
    Epoch  27 Batch    0/33   train_loss = 0.771
    Epoch  28 Batch    0/33   train_loss = 0.707
    Epoch  29 Batch    0/33   train_loss = 0.665
    Epoch  30 Batch    0/33   train_loss = 0.630
    Epoch  31 Batch    0/33   train_loss = 0.587
    Epoch  32 Batch    0/33   train_loss = 0.568
    Epoch  33 Batch    0/33   train_loss = 0.545
    Epoch  34 Batch    0/33   train_loss = 0.510
    Epoch  35 Batch    0/33   train_loss = 0.479
    Epoch  36 Batch    0/33   train_loss = 0.471
    Epoch  37 Batch    0/33   train_loss = 0.446
    Epoch  38 Batch    0/33   train_loss = 0.425
    Epoch  39 Batch    0/33   train_loss = 0.412
    Epoch  40 Batch    0/33   train_loss = 0.392
    Epoch  41 Batch    0/33   train_loss = 0.391
    Epoch  42 Batch    0/33   train_loss = 0.373
    Epoch  43 Batch    0/33   train_loss = 0.366
    Epoch  44 Batch    0/33   train_loss = 0.343
    Epoch  45 Batch    0/33   train_loss = 0.319
    Epoch  46 Batch    0/33   train_loss = 0.313
    Epoch  47 Batch    0/33   train_loss = 0.294
    Epoch  48 Batch    0/33   train_loss = 0.288
    Epoch  49 Batch    0/33   train_loss = 0.270
    Epoch  50 Batch    0/33   train_loss = 0.262
    Epoch  51 Batch    0/33   train_loss = 0.251
    Epoch  52 Batch    0/33   train_loss = 0.244
    Epoch  53 Batch    0/33   train_loss = 0.237
    Epoch  54 Batch    0/33   train_loss = 0.234
    Epoch  55 Batch    0/33   train_loss = 0.230
    Epoch  56 Batch    0/33   train_loss = 0.225
    Epoch  57 Batch    0/33   train_loss = 0.224
    Epoch  58 Batch    0/33   train_loss = 0.223
    Epoch  59 Batch    0/33   train_loss = 0.222
    Epoch  60 Batch    0/33   train_loss = 0.219
    Epoch  61 Batch    0/33   train_loss = 0.219
    Epoch  62 Batch    0/33   train_loss = 0.217
    Epoch  63 Batch    0/33   train_loss = 0.219
    Epoch  64 Batch    0/33   train_loss = 0.217
    Epoch  65 Batch    0/33   train_loss = 0.218
    Epoch  66 Batch    0/33   train_loss = 0.214
    Epoch  67 Batch    0/33   train_loss = 0.215
    Epoch  68 Batch    0/33   train_loss = 0.214
    Epoch  69 Batch    0/33   train_loss = 0.215
    Epoch  70 Batch    0/33   train_loss = 0.215
    Epoch  71 Batch    0/33   train_loss = 0.218
    Epoch  72 Batch    0/33   train_loss = 0.214
    Epoch  73 Batch    0/33   train_loss = 0.214
    Epoch  74 Batch    0/33   train_loss = 0.214
    Epoch  75 Batch    0/33   train_loss = 0.215
    Epoch  76 Batch    0/33   train_loss = 0.213
    Epoch  77 Batch    0/33   train_loss = 0.214
    Epoch  78 Batch    0/33   train_loss = 0.212
    Epoch  79 Batch    0/33   train_loss = 0.213
    Epoch  80 Batch    0/33   train_loss = 0.212
    Epoch  81 Batch    0/33   train_loss = 0.214
    Epoch  82 Batch    0/33   train_loss = 0.212
    Epoch  83 Batch    0/33   train_loss = 0.213
    Epoch  84 Batch    0/33   train_loss = 0.211
    Epoch  85 Batch    0/33   train_loss = 0.213
    Epoch  86 Batch    0/33   train_loss = 0.214
    Epoch  87 Batch    0/33   train_loss = 0.213
    Epoch  88 Batch    0/33   train_loss = 0.213
    Epoch  89 Batch    0/33   train_loss = 0.213
    Epoch  90 Batch    0/33   train_loss = 0.213
    Epoch  91 Batch    0/33   train_loss = 0.212
    Epoch  92 Batch    0/33   train_loss = 0.211
    Epoch  93 Batch    0/33   train_loss = 0.214
    Epoch  94 Batch    0/33   train_loss = 0.211
    Epoch  95 Batch    0/33   train_loss = 0.212
    Epoch  96 Batch    0/33   train_loss = 0.211
    Epoch  97 Batch    0/33   train_loss = 0.212
    Epoch  98 Batch    0/33   train_loss = 0.211
    Epoch  99 Batch    0/33   train_loss = 0.212
    Model Trained and Saved


## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)`


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    inputs = loaded_graph.get_tensor_by_name("input:0")
    initial_state = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    probs = loaded_graph.get_tensor_by_name("probs:0")

    return inputs, initial_state, final_state, probs
```

### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return np.random.choice(list(int_to_vocab.values()), 1, p=probabilities)[0]

```

## Generate TV Script
This will generate the TV script for us.  Set `gen_length` to the length of TV script we want to generate.


```python
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print(tv_script)
```

    moe_szyslak: before you do, i just gotta warn you, marge.(sings) my adeleine....
    crowd:(chanting) barney gumbel" back brace has no turning the eyes.(chuckles) really need cash, homer. nobody loves you...
    c. _montgomery_burns: that was no accident. let's get out of here.
    moe_szyslak:(counting radishes) now where you wanna know, we make this the life?
    homer_simpson: oh, thank god, the gimmick of.(to barflies) shut up, moe.
    moe_szyslak: come on, you greedy old reptile!
    c. _montgomery_burns:(shaken) why not my friends! homer didn't buy!
    seymour_skinner:(holding two-thirds-empty beer) two of us have drink a new lease on" alcohol half it.
    homer_simpson: i'm gonna treat marge to a romantic dinner.
    moe_szyslak: ooh, taking advantage, here.(warmly) one" flaming homer" the sign, barney and easygoing.
    lenny_leonard: i was just why not in here no more, homer


# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.
