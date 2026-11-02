# Machine Learning II: course notebooks

Jupyter notebooks I build class by class while taking **Aprendizaje Automático II** (Machine Learning II). Each one starts from the class material and adds the exercises I solved: comparing models, tuning hyperparameters, plotting results and drawing conclusions.

The path goes from a single perceptron to a local LLM with LoRA and RAG.

> The notebooks' comments and explanations are in Spanish.

| # | Notebook | Topic | Dataset | Highlight |
|---|----------|-------|---------|-----------|
| 1 | [`1_Perceptron.ipynb`](1_Perceptron.ipynb) | Perceptron and MLP | Wine (sklearn) | 3-layer MLP, 3 classes: 97.8% test accuracy |
| 2 | [`2_Word2vec.ipynb`](2_word2vec.ipynb) | Word2Vec and word embeddings | IMDb reviews | Sentiment from averaged embeddings: ~84% test accuracy |
| 3 | [`3_CNN.ipynb`](3_CNN.ipynb) | Convolutional neural networks | MNIST | 2-layer CNN: 98.8% test accuracy |
| 4 | [`4_Texto.ipynb`](4_Texto.ipynb) | Recurrent networks for text (RNN, LSTM, GRU) | AG News | GRU: 91.9% test accuracy (4 classes) |
| 5 | [`5_GNN.ipynb`](5_GNN.ipynb) | Graph neural networks | Cora, CiteSeer, PubMed | GAT: 82.3% test accuracy on Cora |
| 6 | [`6_LLMs.ipynb`](6_LLMs.ipynb) | LLMs in practice (local model) | MMLU + custom data | RAG: from 0/5 to 4/5 correct answers |

---

## 1. Perceptron and MLP: `1_Perceptron.ipynb`

**Concepts:** linear neuron + sigmoid, binary cross-entropy, gradient descent (SGD), backpropagation, hidden layers with ReLU, multiclass classification with `CrossEntropyLoss` and `argmax`.

**What's in it**
- Binary classification (Wine classes 0 and 1) with scikit-learn's logistic regression as a baseline.
- The same problem with a perceptron in PyTorch, writing the training loop by hand.
- **Exercise:** a 3-layer dense network (`MLP3`) compared against the perceptron.
- **Extra:** the full 3-class problem, with 3 output logits and `CrossEntropyLoss`.

**Results:** all three binary models reach ~97% test accuracy, and the multiclass MLP reaches 97.8%. The test set has only 33 samples, so the models differ by a single sample.

Theory notes for this class and the next one (in Spanish): [`resumen_sesion_mlp_v2v.md`](resumen_sesion_mlp_v2v.md).

## 2. Word2Vec: `2_Word2vec.ipynb`

**Concepts:** word embeddings, Skip-gram vs CBOW, negative sampling, cosine similarity, vector analogies, document vectors as averaged embeddings, t-SNE.

**What's in it**
- Word2Vec (gensim) trained on IMDb reviews, exploring similarities and analogies (`james - cameron + bruce → willis`).
- Each review becomes the average of its word embeddings, and a logistic regression classifies its sentiment.
- **Exercise 1:** Skip-gram vs CBOW, plus sweeps over `window`, `vector_size` and `negative`.
- **Exercise 2:** 2D visualization of the embeddings with t-SNE.
- **Exercise 3:** one Word2Vec trained only on positive reviews and another only on negative ones, comparing word similarities.

**Results:** Skip-gram 83.1% vs CBOW 81.4% test accuracy. The best run in the sweep was `vector_size=175` (84.0%).

## 3. Convolutional neural networks: `3_CNN.ipynb`

**Concepts:** convolution, max pooling, flatten, fully connected layer, Adam, mini-batches with `DataLoader`, learning curves.

**What's in it**
- A CNN with 2 convolutional layers to classify MNIST digits.
- **Exercise 1:** loss and accuracy learning curves (train vs test).
- **Exercise 2:** 1 to 3 convolutional layers with a `FlexCNN` class, averaging 3 seeds and measuring parameters and training time.
- **Exercise 3:** binarized (black/white) images vs grayscale.

**Results**

| Conv layers | Parameters | Test accuracy (mean ± std) |
|---|---|---|
| 1 | 15,770 | 97.25 ± 0.17% |
| 2 | 9,098 | 98.32 ± 0.14% |
| 3 | 8,778 | 98.45 ± 0.08% |

More layers give better accuracy **with fewer parameters**, because pooling shrinks the final dense layer. With binarized images the CNN reaches 98.6%, about the same as with grayscale.

## 4. Recurrent networks for text: `4_Texto.ipynb`

**Concepts:** tokenization and vocabulary, `<pad>`/`<unk>`, padding and truncation, `nn.Embedding`, `pack_padded_sequence`, RNN, LSTM, GRU.

**What's in it**
- A full text pipeline: custom vocabulary, `Dataset` and `collate_fn` to classify AG News articles into 4 categories.
- An LSTM classifier trained from scratch.
- **Exercise 1:** learning curves.
- **Exercise 3:** simple RNN vs LSTM vs GRU, with the same seed and hyperparameters.

**Results**

| Model | Test accuracy |
|---|---|
| RNN | 89.3% |
| LSTM | 91.5% |
| GRU | 91.9% |

## 5. Graph neural networks: `5_GNN.ipynb`

**Concepts:** node classification, message passing, GCN, GraphSAGE, GAT (attention over neighbors), early stopping and model selection on validation.

**What's in it**
- A 2-layer GCN on the Cora citation graph, with PyTorch Geometric.
- **Exercise 1:** early stopping, restoring the best epoch by validation accuracy.
- **Exercise 2:** GCN vs GraphSAGE vs a hand-written message passing layer vs GAT.
- **Exercise 3:** the same GCN on Cora, CiteSeer and PubMed.

**Results (Cora, test):** GAT 82.3% · Message Passing 81.1% · GraphSAGE 80.0% · GCN 79.9%.

## 6. LLMs in practice: `6_LLMs.ipynb`

Everything runs on CPU with a small LLM (`Qwen2.5-0.5B-Instruct`).

**Concepts:** decoder that predicts the next token, temperature, MMLU-style evaluation, pass@k vs pass^k, where the parameters live (attention vs MLP), near-orthogonality in high dimensions, LoRA, sensitive-data sanitization (p-filtering), LLM limitations, RAG.

**What's in it**
- Next-token distribution and the effect of temperature.
- MMLU (high school computer science) scored by the probability of each answer letter: 37.5% vs 25% chance.
- pass@k and pass^k on problems with numeric answers.
- Parameter breakdown of the model, and BERT vs RoBERTa (orthogonality of weights).
- **LoRA** fine-tuning (0.11% trainable parameters) to teach the model a new fact.
- Redacting and substituting low-probability words with BERT to anonymize text.
- Limitations: cross-language consistency, system-prompt rules that aren't followed, compounding errors.
- **RAG** with multilingual embeddings and the *"power of noise"* experiment: how distractor documents and the position of the right document affect answers.

---

## Stack

Python · PyTorch · scikit-learn · gensim · torchvision · PyTorch Geometric · Hugging Face (`transformers`, `datasets`, `peft`, `sentence-transformers`) · matplotlib

## How to run

Requires Python 3.12 or newer (some f-strings use nested quotes, which need 3.12).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # in CMD: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook
```

Notes:
- Datasets download automatically the first time: MNIST to `dataset/`, Planetoid to `data/`, and IMDb, AG News and MMLU to the Hugging Face cache.
- `6_LLMs.ipynb` downloads about 3 GB of models the first time.
- `4_Texto.ipynb` takes about 20 minutes per model on CPU.

## Credits

The base notebooks and exercise statements are course material from the Aprendizaje Automático II teaching staff. The exercise solutions, comparisons and conclusions are my own.
