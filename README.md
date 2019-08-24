# heavywater
# mortgageDocPredictor

This project is in two parts - code in the `training` folder to train a tensorflow/keras MLP model locally using vectorized n-grams - and code in the `prediction` folder to host the model in a flask app that is deployable to AWS using the Elastic Beanstalk cli.

Note - in hindsight, this project could be improved by training the model on ec2 during initialization of the webapp, removing the necessity of pickling objects and uploading the model.

Framing the training problem:

This is a document classification problem. Following guidelines found in this guide https://developers.google.com/machine-learning/guides/text-classification/ , after exploring the data, given the relatively small `number of samples/number of words per sample ratio`, a Multi Layer Perceptron model was chosen with the strategy of splitting the words into n-grams, vectorizing the n-grams, and selecting the top 20k scoring n-grams for use in training. 

To reproduce the model:

- `cd` into training dir
- `python3 -m venv virt` - create virtualenv
- `source virt/bin/activate` - activate virtualenv
- `pip3 install -r requirements.txt` - install training dependencies
- `python3 train.py` - cleans the data removing rows that are missing labels or words and trains a new model, which is saved to `../prediction/model` dir. Also saves pickles of label decoder, vectorizer, and selector objects needed later for making predictions to the `../prediction/pickles` dir


Framing the deployment problem:

As the code for making predictions requires large libraries and relatively large memory space during operation, the strategy of hosting the model in a flask app deployed to ec2 on an r5.large was chosen (this may be overkill but a t2.micro did not have enough memory, and initial attempts to deploy to lambda via sam cli failed due to lack of disk space on lambda). Elastic Beanstalk was chosen as the managed service to deploy the app to ec2 as it generates the auxiliary resources needed for the user.

To reproduce the deployment:

- `cd` into prediction dir
- `python3 -m venv virt` - create virtualenv
- `source virt/bin/activate` - activate virtualenv
- `pip3 install -r requirements.txt` - install webapp/prediction dependencies
- install eb cli - this is `brew install awsebcli` on mac
- `eb init -p python-3.7 mortgage-doc-predictor --region us-east-1` - initializes new elastic beanstalk project
- `eb config` - this opens elastic beanstalk configs in an editor - change `InstanceType` to `r5.large` and `WSGIPath` to `app.py`
- `eb create mortgage-doc-predictor-env` - begins creating of aws resources and deployment of app
- `eb open` - once deployment is finished and ec2 is running and flask app is initialized, opens app in browser

App can be run locally with `flask run` and visited at `localhost:5000`

Currently running on AWS at http://mortgage-doc-predictor-env.6qukfteaus.us-east-1.elasticbeanstalk.com/ 


