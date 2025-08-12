# **Naive Bayes**

## Downloading and Splitting data

importing needed libraries:

```python
from utils import process_tweet, lookup
import pdb # Python’s built-in debugger.
           # Use pdb.set_trace() to start debugging from any point in your code. 
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
```

Splitting downloaded data:

```python
# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
# 'positive_tweets.json' is the name of a file inside the twitter_samples corpus that
# contains positive tweets.
# twitter_samples.strings() loads all tweets from that file as a list of strings you can
# use directly.

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))
```

## **Implementing helper functions**

We will define a count_tweets function similar to the previous one, build_freqs.

### count_tweets

```python
def count_tweets(result, tweets, ys):
    """
    Update a dictionary counting occurrences of (word, sentiment) pairs.

    Args:
        result: dict
            Existing dictionary mapping (word, sentiment) tuples to frequencies.
        tweets: list of str
            List of tweet texts.
        ys: list of int
            List of sentiment labels corresponding to each tweet (0 or 1).

    Returns:
        dict
            Updated dictionary with frequencies of (word, sentiment) pairs.
    """

    # Iterate over each tweet and its corresponding sentiment label
    for y, tweet in zip(ys, tweets):
        # Process the tweet text into individual words (e.g., lowercase, tokenize)
        for word in process_tweet(tweet):
            # Create a key combining the word and its sentiment label
            pair = (word, y)

            # If the pair is already in the dictionary, increment its count
            if pair in result:
                result[pair] += 1
            # Otherwise, add the pair with an initial count of 1
            else:
                result[pair] = 1

    # Return the updated dictionary of frequencies
    return result
```

### difference between `count_tweets` and `build_freqs`

**`count_tweets`** updates an **existing** frequency dictionary you pass in (useful for incremental updates), but **`build_freqs`** creates a **new** frequency dictionary from scratch each time it’s called. 
Also, `build_freqs` handles converting `ys` from a NumPy array to a list, while `count_tweets` assumes `ys` is already a list.

## **Training model using Naive Bayes**

### 1. Prior probabilities (how common each class is)

The first part of training a naive bayes classifier is to identify the number of classes that you have. We will create a probability for each class.

$D_{pos}$ = number of positive tweets

$D_{neg}$ = number of negative tweets

$D$ = total tweets 

probability for each class:

$$
P(D_{pos}) = \frac{D_{pos}}{D}\tag{1}
$$

$$
P(D_{neg}) = \frac{D_{neg}}{D}\tag{2}
$$

### 2. Logprior (bias toward positive or negative)

We don’t just store the two probabilities — we store their ratio in **log** form:  

$$
\text{logprior} = log \left( \frac{P(D_{pos})}{P(D_{neg})} \right) = log \left( \frac{D_{pos}}{D_{neg}} \right)

$$

Or equivalently:

$$
\text{logprior} = \log (P(D_{pos})) - \log (P(D_{neg})) = \log (D_{pos}) - \log (D_{neg})\tag{3}
$$

**Positive value** → dataset has more positive tweets
**Negative value** → dataset has more negative tweets

Why log? Because when we combine probabilities for many words later, adding logs is much easier (and avoids tiny numbers).

### 3. Positive and Negative Probability of a Word (likelihoods)

$freq_{pos}$ =  how many times this word appears in positive tweets

$freq_{neg}$ = how many times this word appears in negative tweets

$N_{pos}$ = total words in **all** positive tweets(counting duplicates)

$N_{neg}$
 = total words in **all** negative tweets(counting duplicates)

$V$ = vocabulary size (unique words total)

positive(`P(word | positive)`) and negative(`P(word | negative)`) probability for a specific word W:

$$
P(W_{pos}) = \frac{freq_{pos} + 1}{N_{pos} + V}\tag{4}

$$

$$
P(W_{neg}) = \frac{freq_{neg} + 1}{N_{neg} + V}\tag{5}
$$

**Additive smoothing** (the +1) ensures no word gets a probability 0 just because it didn’t appear in one class.

### Where this formula comes from?

That formula comes directly from the **multinomial Naive Bayes** model with **additive smoothing** (also called Laplace smoothing).

**1. Base idea: Naive Bayes conditional probability**

“If the tweet is positive, what’s the probability of seeing this word?”:

$$
P(word∣class)

$$

**2. Without smoothing**

The straightforward estimate from counts is:

$$
P(\text{word} \mid \text{positive}) = \frac{\text{count of the word in positive tweets}}{\text{total number of words in positive tweets}} = \frac{freqs[(word,1)]}{N_{pos}}
$$

1. **Additive smoothing**

If a word never appeared in positive tweets during training, its numerator is 0, so:

$$
P(word∣positive)=0
$$

And Naive Bayes multiplies probabilities together when scoring a document. One zero wipes out the whole product →**disaster**.
We fix this by adding 1 to *every* word count:

$$
P(word∣positive)=\frac{count+1}{\text{total words}+V}

$$

`+1` in numerator → gives unseen words a small nonzero probability
`+V` in denominator → keeps the distribution normalized (we added 1 for each of V possible words)

### 4. Loglikelihood

Finally, for each word, we measure **how much it leans positive vs negative**:

$$
\text{loglikelihood} = \log \left(\frac{P(W_{pos})}{P(W_{neg})} \right)\tag{6}

$$

**loglikelihood > 0** → word is more common in positive tweets
**loglikelihood < 0** → word is more common in negative tweets
The further from 0, the stronger the association.

## Implementing a Naive Bayes Classifier in Code

### **Creating `freqs` dictionary**

using `count_tweets()` function that we created earlier. We will use **`freqs`** dictionary in several parts of code.

```python
# Build the freqs dictionary for later uses

freqs = count_tweets({}, train_x, train_y)
```

### Creating `logprior, loglikelihood` (Naive Bayes Main Code)

```python
def train_naive_bayes(freqs, train_x, train_y):
    """
    Train a Naive Bayes classifier.

    Args:
        freqs: dict mapping (word, label) -> frequency
        train_x: list of tweets (unused here, but kept for API consistency)
        train_y: list/array of labels (0 = negative, 1 = positive)

    Returns:
        logprior: float, log prior probability
        loglikelihood: dict mapping word -> log likelihood ratio
    """
    # Vocabulary size
    vocab = {word for word, _ in freqs.keys()}
    # That’s set comprehension. It’s just a fancy way to:
    # Loop over every (word, label) pair in the keys.
    # Keep only the word part (word).
    # Store them in a set, which automatically removes duplicates.
    # sets are perfect when you want to find all unique items from a list or iterable.
    V = len(vocab)

    # Total word counts in positive and negative tweets
    N_pos = sum(freq for (word, label), freq in freqs.items() if label == 1)
    N_neg = sum(freq for (word, label), freq in freqs.items() if label == 0)
    # freq for ... if ... is a generator expression.
    # We used a generator expression here to efficiently sum frequencies without creating
    # an intermediate list, saving memory and keeping the code concise.

    # Number of documents
    D_pos = np.sum(np.array(train_y) == 1)
    D_neg = np.sum(np.array(train_y) == 0)
    # np.array(train_y) converts the list train_y (which contains labels like 0 or 1 for
    # each tweet) into a NumPy array. This allows easy element-wise operations.
    # np.array(train_y) == 1 creates a Boolean array where each element is True if the
    # corresponding label is 1 (positive), else False.

    # Log prior
    logprior = np.log(D_pos) - np.log(D_neg)
    # If D_pos > D_neg, this value is positive, meaning the dataset has more positives.

    # Log likelihood for each word
    loglikelihood = {
        word: np.log((lookup(freqs, word, 1) + 1) / (N_pos + V)) -
              np.log((lookup(freqs, word, 0) + 1) / (N_neg + V))
        for word in vocab
    }
    # a dictionary comprehension: a shortcut way of building a dictionary in one line,
    # instead of:
    # 1.Creating an empty dictionary
    # 2.Using a loop to fill it
    # general pattern: {key: value for item in something}
    # similar to:
    # loglikelihood = {}
    # for word in vocab:
         # p_w_pos = (lookup(freqs, word, 1) + 1) / (N_pos + V)
         # p_w_neg = (lookup(freqs, word, 0) + 1) / (N_neg + V)
         # loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)

    return logprior, loglikelihood
```

### `lookup()` in utils

This gets how many times `word` appears in tweets with the given `label`.

```python
def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n
```

## **Test Our Naive Bayes**

### **Implement `naive_bayes_predict`**

The function takes in the `tweet`, `logprior`, `loglikelihood`. 
It returns the probability that the tweet belongs to the positive or negative class. 
For each tweet, sum up the loglikelihoods of each word in the tweet. 
Additionally, add the logprior to this sum to obtain the predicted sentiment of that tweet.

$$
p = logprior + \sum_i^N (loglikelihood_i)
$$

If the dataset is balanced, logprior = 0 and doesn’t affect results, but we keep it for unbalanced cases.

```python
def naive_bayes_predict(tweet, logprior, loglikelihood):
    """
    Predict sentiment score for a tweet using a trained Naive Bayes model.

    Args:
        tweet (str): The tweet text.
        logprior (float): The log prior (overall bias toward positive or negative sentiment).
        loglikelihood (dict): Mapping from word -> log likelihood ratio.

    Returns:
        float: Sentiment score. 
               >0 means positive sentiment, <0 means negative sentiment.
    """
    word_list = process_tweet(tweet)

    # Start the score (probability in log form) with the log prior
    # This represents the overall bias from the training data
    p = logprior

    # Loop through each processed word in the tweet
    for word in word_list:
        # If the word exists in our trained loglikelihood dictionary
        if word in loglikelihood:
            # Add its log likelihood score to p
            # (If the word is more common in positive tweets, this increases p; 
            # if more common in negative tweets, it decreases p)
            p += loglikelihood[word]

    # Return the final log probability score
    return p
    # instead of loop we can use a generator expression:
    # return logprior + sum(loglikelihood[word] for word in word_list if word in loglikelihood)
```

```python
# Experiment with your own tweet.
my_tweet= 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)
```

### **Implement `test_naive_bayes`**

The function takes in `test_x`, `test_y`, `log_prior`, and `loglikelihood`.
It returns the accuracy of your model.
First, we use `naive_bayes_predict` function to make predictions for each tweet in text_x.

```python
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Evaluate the accuracy of the Naive Bayes classifier on test data.

    Args:
        test_x (list of str): List of tweets to classify.
        test_y (list or array): True sentiment labels (0 or 1) for each tweet.
        logprior (float): Log prior probability from training.
        loglikelihood (dict): Dictionary mapping words to their log likelihood ratios.

    Returns:
        float: Accuracy = proportion of correctly predicted tweets.
    """

    # Step 1: Predict labels for each tweet
    # Use a list comprehension to apply naive_bayes_predict on every tweet
    # If predicted score > 0, classify as positive (1), else negative (0)
    y_hats = [
        1 if naive_bayes_predict(tweet, logprior, loglikelihood) > 0 else 0
        for tweet in test_x
    ]

    # Step 2: Compare predicted labels with true labels
    # Convert both lists to numpy arrays for element-wise comparison
    y_hats_array = np.array(y_hats)
    test_y_array = np.array(test_y)

    # Step 3: Calculate accuracy as the fraction of correct predictions
    accuracy = np.mean(y_hats_array == test_y_array)

    # Step 4: Return the accuracy score
    return accuracy
    
    # instead of list comprehension:
    #y_hats = []
    #for tweet in test_x:
        #if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            #y_hat_i = 1
        #else:
            #y_hat_i = 0
        #y_hats.append(y_hat_i)
```

## Error Analysis

To see which tweets our model misclassified.

```python
# Print header row for clarity
print(f"{'Truth':<6} {'Predicted':<10} Tweet")
# <6 and <10 in f-strings → Align columns neatly for readability.
# :<6 → left-align the value in a field that’s 6 characters wide.
# :<10 → left-align in a field 10 characters wide.

# Go through each tweet and its true label
for tweet, true_label in zip(test_x, test_y):
    # Get the Naive Bayes prediction score (sum of log likelihoods + logprior)
    score = naive_bayes_predict(tweet, logprior, loglikelihood)
    
    # Convert the score into a predicted label:
    # If score > 0 → predict 1 (positive), else 0 (negative)
    predicted_label = int(score > 0)
    # Converts a boolean to integer
    # int(True) → 1
    # int(False) → 0
    
    # If the model got this tweet wrong, show it for error analysis
    if predicted_label != true_label:
        # Process tweet into a cleaned list of words
        processed_words = process_tweet(tweet)
        
        # Join words back into a string
        processed_text = ' '.join(processed_words)
        # .join(processed_words) takes all the items in the list and glues them together
        # into one single string, placing the separator (' ') between each item.
        
        # Ensure it's ASCII-safe (remove emojis and non-English chars)
        safe_text = processed_text.encode('ascii', 'ignore').decode()
        # If a character can’t be represented in ASCII (like 😍 or é), just skip it — don’t crash.
        # .decode():This takes the bytes (b'...') and turns them back into a normal Python string.
        
        # Print the true label, predicted label, and the processed tweet
        print(f"{true_label:<6} {predicted_label:<10} {safe_text}")

```
