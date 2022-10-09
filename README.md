# Model Architecture

The proposed self-supervised clustering technique employs the following architecture based on an LSTM language model:

[](cluster-arch.svg)

## Components

- **S**: Token-to-type embedding. $S: \\{0, 1\\}^n \mapsto \mathbb{R}^s$ is an Embedding layer which maps one-hot token ids 
  (labeled $t_1$ through $t_n$) onto *s*-dimensional weight vectors. Each weight represents the likelihood that the token 
  in question belongs to one of $s$ types. Each type is interpreted as representing a distinct sign name. The subsequent 
  softmax layer converts these weight vectors into probability distributions over types/sign names.
- **E**: Type embedding. $E: \mathbb{R}^s \mapsto \mathbb{R}^d$ is also a standard Embedding layer. When the input to $E$ is 
  a one-hot vector, the output is a dense *d*-dimensional embedding for the corresponding type/sign name. If the input is not 
  one-hot, the output is a weighted sum over the *d*-dimensional type/sign name embeddings.

# Training Objective

This model is trained to minimize a combination of three loss terms: 
$$\mathcal{L} = \mathcal{L}\_{hom} + \mathcal{L}\_{LM} + \mathcal{L}\_{dup}$$

## Homogeneity Loss
$$\mathcal{L}\_{hom} = \frac{1}{s} \frac{1}{n} \sum_{j=1}^s \sum_{i=1}^{n} p_{ij}d(i, \textrm{ex}(j))$$

where $p\_{ij}$ is the probability that token $t_i$ is assigned to type/sign name $j$ and 
$\textrm{ex}(j) = \textrm{argmax}\_{i} p\_{ij}$ identifies the "exemplar" token 
$t\_{\textrm{ex}(j)}$ which is most strongly associated with sign $j$. 
$d(u, v)$ measures the visual dissimilarity between images of tokens $t\_u$ and $t\_v$.

For each type/sign, this term identifies a token to use as the exemplar for that sign. 
It then computes the visual similarity of every token to the exemplar, weighted by the likelihood that the token is an instance of this sign.
This penalizes assignments which label visually-distinct tokens as instances of the same sign, and compels the model to reduce $p_{ij}$ when $t_i$ does not resemble the exemplar for sign $j$.

$d(u,v)$ can be implemented using any function whose value grows with the degree of visual difference between $t_u$ and $t_v$.
We use the L1 norm of $(\textrm{img}_u-\textrm{img}_v)$ where $\textrm{img}_u$ and $\textrm{img}_v$ are images (that is, matrices of pixel intensities) of $t_u$ and $t_v$.

## LM Loss
$$\mathcal{L}\_{LM} = CE( o, t )$$

where $CE$ is categorical cross-entropy loss. $o$ is the sequence of outputs from the LSTM. The sequence of target labels, $t$, is equal to the sequence of outputs from $S$ (which are distributions over types/sign names) shifted by one time-step.
This means that the LSTM is trained to recover the types/sign names predicted by the token-to-type embedding layer $S$.
This self-supervised configuration is intended to guarantee that each sign represents contextually similar tokens, as combining unrelated tokens under the same label will increase the uncertainty of the following signs and decrease LM performance.

## Duplicate Cluster Loss
$$\mathcal{L}\_{dup} = \sum\_{j=1}^s \max\_{k=j+1}^s \left( 1-d(ex(j), ex(k))\right)\left(\frac{1}{n} \sum\_{i=0}^n p\_{ij}\right)$$

For each type/sign name $j$, this term first computes the mean likelihood $\frac{1}{n} \sum\_{0 \leq i \leq n} p_{ij}$ that any 
token has been labeled with this sign. This is the *usage penalty*, which will be larger the more tokens belong to this sign.
It next computes a *similarity penalty*, which equals the maximum similarity between this sign's exemplar and subsequent 
signs' exemplars. The overall loss for each sign is the product of the usage penalty and the similarity penalty.

This has the effect of removing duplicate signs. When two signs represent similar sets of tokens, their exemplars will be 
visually similar and the similarity penalty will be large. When this happens, the model is encouraged to reduce every 
token's association with the first of the duplicate signs. This results in these tokens being relabeled until the 
duplicate sign becomes unused or its exemplar changes.
