Hey there, really excited for you to take a look through the code and any questions that you have. 

This code takes recipes.csv and similarity_scores.csv (or explicit file paths) and produces a weighted hybrid 
similarity matrix of all recipes.

Features of this code include:
* Enforcing of a data contract for input data
* Limited unit tests (these would be more specific and expanded for actual production)
* Basic data cleaning and transformation
* Produces multiple content based similarities from recipes_info. These include
    * tfidf or count vectorisation
    * cosine similarity or jaccard similarity
    * standard or customer tokenisation of fields
* Produces a single cosine similarity of scored data
* Produces a weights hybrid similarity score summarised: 
<a href="https://www.codecogs.com/eqnedit.php?latex=hybridSim&space;=&space;\alpha&space;\times&space;simCF_{ij}&space;&plus;&space;(1&space;-&space;\alpha)&space;\times&space;simCB_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?hybridSim&space;=&space;\alpha&space;\times&space;simCF_{ij}&space;&plus;&space;(1&space;-&space;\alpha)&space;\times&space;simCB_{ij}" title="hybridSim = \alpha \times simCF_{ij} + (1 - \alpha) \times simCB_{ij}" /></a>


```
# Basic usage:
from recipe_similarities.similarities import SimilarityFactory
sf = SimilarityFactory()
sf.load_data()
hsims = sf.hybrid_similarities()

hsim returns dict {'similarity_matrix': NxN matics as a numpy array,
                   'index': Pandas Index object highlightin recipe IDs to cross reference with the similarity matrix}
```

hybrid_similarities() is parameterised, feel free to adjust these parameters to alter the output and experiment.

```
:param alpha: this adjusts the weighting 1 = pure scored similarities only,
                                         0 = pure content similarities only,
                                         In between 0 - 1 weights either side
:param cb_method: accepts 'cosine' or 'jaccard'
:param cbf_method: accepts 'cosine' only
:param text_vectoriser: accepts 'tfid' or 'count'
:param strict: True default. This enforces the tokenisation where each field's whole string is a token.
              False. This uses a default tokeniser to compare each recipes as if it were a single text blob.
```

Looking forward to meeting you, Al. 