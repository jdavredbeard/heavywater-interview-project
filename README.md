# heavywater
# mortgageDocPredictor

This project is in two parts - code in the `training` folder to train a tensorflow/keras MLP model using vectorized n-grams - and code in the `prediction` folder to host the model in a flask app that is deployable to AWS using the Elastic Beanstalk cli.

Framing the problem:

This is a document classification problem. Following guidelines found in this guide https://developers.google.com/machine-learning/guides/text-classification/ , after exploring the data, given the relatively small `number of samples/number of words per sample ratio`, a Multi Layer Perceptron model was chosen with the strategy of splitting the words into n-grams, vectorizing the n-grams, and selecting the top 20k scoring n-grams for use in training.

To reproduce the existing deployed app:

Training Steps:

- `cd` into training dir
- `python3 -m venv virt` - create virtualenv
- `pip3 install -r requirements.txt` - install training dependencies
- `python3 train.py` - cleans the data removing rows that are missing labels or words and trains a new model, which is saved to `../prediction/model` dir. Also saves pickles of label decoder, vectorizer, and selector objects needed later for making predictions to the `../prediction/pickles` dir
- 

