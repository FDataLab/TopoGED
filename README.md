# TOPOGED: A Topological Encoder Decoder Framework For Temporal Graph Learning

A Temporal Graph Generation Method, powered by TopER<sup>1</sup>. Our goal is to construct a series of temporal graphs, given prior history and the current predicted TopER vector. We aim to construct graphs that have nodes that appear and dissapear over time.

## Probability Counts

We present a way to tell how many edges of each type will be added to a graph, and how many old and new nodes will appear. 

Edge Types:
> - o-o-bank: An edge, between two old nodes, that has previously had an edge
> - o-o-nobank: An edge, between two old nodes, that has not previously had an edge
> - o-n: An edge, between one new node and one old node
> - n-n: An edge, between two new nodes

TODO Update the terminology here

## Proposed Methods

All methods take, at minimum, a TopER vector and a probability count vector as input to guide their construction.

- Purely Greedy Algorithm:
> A purely random method, it does not account for the degree limitations of TopER vectors.
> Old nodes can reappear in new graphs, and new nodes are assigned the next available ID.
> File Location: `GraphGeneration/scripts/random_gen_contids.py`
> Arguments:
>   - --dataset: The dataset to run on. If it does not exist in `data/input/cached/` already, it will be processed if it is a valid edgelist

- Greedy, Leverages TopER:
> A purely random method, but it restricts nodes to have some maximum degree as dictated by TopER
> Old nodes can reappear in new graphs, and new nodes are assigned the next available ID.
> File Location: `GraphGeneration/scripts/random_gen_contids_degree.py`
> Arguments:
>   - --dataset: The dataset to run on. If it does not exist in `data/input/cached/` already, it will be processed if it is a valid edgelist
>   - --oldDegree: Whether or not reappearing nodes should retain their old degree (For example, if Node 1 had a degree of 3 in its most recent appearance, it will have maximum of 3 in this graph)

- Supervised Learning, One Training:
> Using supervised learning, we predict what edges will happen in a graph given positive/negative samples from previous graphs.
> The MLP used is trained one time, on a set number of graphs. Then is used to predict for all graphs
> By embedding nodes from previous graph structure, we use an MLP to predict the probability of an edge between two nodes. Then we choose the most likely edges (as many are necessary)
> There are many arguments that customize this strategy, and outputs will vary
> File Location: `GraphGeneration/scripts/gen_with_model.py`
> Arguments:
>   - --dataset: The dataset to run on. If it does not exist in `data/input/cached/` already, it will be processed if it is a valid edgelist
>   - --strategy: What type of MLP to use. Either a single MLP that is shared for all edge types or a MultiHeadedMLP that has one head per edge type.
>   - --embedding: If you want to add on positional encodings or node types to the embedding
>   - --mlpEncoding: How you want to feed node embeddings into the MLP
>   - --embedOld: Whether or not you want to let the MLP predict the edge type 'o-o-bank' or let them be randomly added
>   - --oldDegree: Whether or not reappearing nodes should retain their old degree (For example, if Node 1 had a degree of 3 in its most recent appearance, it will have maximum of 3 in this graph)
>   - --trainingStyle: Which graphs you want to provide to the MLP. Either only the true graphs, only the predicted, or a mix.
>   - embeddingType: How you want to embed nodes, either with Node2Vec<sup>2</sup> or with a Linear computation inspired by GraphAny<sup>3</sup>.

- Supervised Learning, Retrains On Each New Graph:
> Using supervised learning, we predict what edges will happen in a graph given positive/negative samples from previous graphs.
> Before we construct a graph at timestep *t*, we train the MLP on positive/negative samples from graphs *0-(t-1)*
> By embedding nodes from previous graph structure, we use an MLP to predict the probability of an edge between two nodes. Then we choose the most likely edges (as many are necessary)
> There are many arguments that customize this strategy, and outputs will vary
> File Location: `GraphGeneration/scripts/gen_with_model_retrain.py`
> Arguments:
>   - --dataset: The dataset to run on. If it does not exist in `data/input/cached/` already, it will be processed if it is a valid edgelist
>   - --strategy: What type of MLP to use. Either a single MLP that is shared for all edge types or a MultiHeadedMLP that has one head per edge type.
>   - --embedding: If you want to add on positional encodings or node types to the embedding
>   - --mlpEncoding: How you want to feed node embeddings into the MLP
>   - --embedOld: Whether or not you want to let the MLP predict the edge type 'o-o-bank' or let them be randomly added
>   - --oldDegree: Whether or not reappearing nodes should retain their old degree (For example, if Node 1 had a degree of 3 in its most recent appearance, it will have maximum of 3 in this graph)
>   - embeddingType: How you want to embed nodes, either with Node2Vec<sup>2</sup> or with a Linear computation inspired by GraphAny<sup>3</sup>.

- Reinforcement Learning
> A previously tested, but scrapped method
> By using Reinforcement Learning, specifically PPO Agents, the hope was to train an Agent that can construct graphs given its previous experiences
> There were many methods, and a new environment for each one. See `ReinforcementLearning/scripts/reconstruction.py` or `ReinforcementLearning/scripts/reconstruction_nx.py` to test.
> See `ReinforcementLearning/reinforcement_utils/adj_envs/` or `ReinforcementLearning/reinforcement_utils/nx_envs/` to see available methods

## References
1. Tola, A., Taiwo, F. M., Akcora, C. G., & Coskunuzer, B. (2024). *TopER: Topological Embeddings in Graph Representation Learning*. arXiv preprint. [https://arxiv.org/abs/2410.01778](https://arxiv.org/abs/2410.01778)  
2. Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks*. arXiv preprint. [https://arxiv.org/abs/1607.00653](https://arxiv.org/abs/1607.00653)  
3. Zhao, J., Zhu, Z., Galkin, M., Mostafa, H., Bronstein, M., & Tang, J. (2025). *Fully-inductive Node Classification on Arbitrary Graphs*. arXiv preprint. [https://arxiv.org/abs/2405.20445](https://arxiv.org/abs/2405.20445)
