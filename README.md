# mortgageDocPredictor

This project is in two parts - code in the `training` folder to train a tensorflow/keras MLP model locally using vectorized n-grams - and code in the `prediction` folder to host the model in a flask app that is deployable to AWS using the Elastic Beanstalk cli.

Framing the training problem:

This is a document classification problem. Following guidelines found in this guide https://developers.google.com/machine-learning/guides/text-classification/ , after exploring the data, given the relatively small `number of samples/number of words per sample ratio`, a Multi Layer Perceptron model was chosen with the strategy of splitting the words into n-grams, vectorizing the n-grams, and selecting the top 20k scoring n-grams for use in training. 

To reproduce the model:

- `cd` into training dir
- `python3 -m venv virt` - create virtualenv
- `source virt/bin/activate` - activate virtualenv
- `pip3 install -r requirements.txt` - install training dependencies
- `python3 train.py` - cleans the data removing rows that are missing labels or words and trains a new model, which is saved to `../prediction/model` dir. Also saves pickles of label decoder, vectorizer, and selector objects needed later for making predictions to the `../prediction/pickles` dir

Testing the model:

The best accuracy over the whole dataset that I've been able to produce so far is about 87%. However, as the class distribution plot shown at the end of the training script illustrates, the class frequency is not evenly distributed, so the model will most likely be better at recognizing a BILL or POLICY CHANGE than an INTENT TO CANCEL NOTICE.
Simple testing of the model seems to bear this out.

- `deactivate` - deactivate training virtualenv,
- `cd` into prediction dir
  - `python3 -m venv virt` - create virtualenv
- `source virt/bin/activate` - activate virtualenv
- `pip3 install -r requirements.txt` - install webapp/prediction dependencies
- `python3 tests.py` - run tests against `get_prediction()`

More thorough testing would include a larger sample of each document class to get an estimate of the accuracy of the model on each class.

Framing the deployment problem:

As the code for making predictions requires large libraries and relatively large memory space during operation, the strategy of hosting the model in a flask app deployed to ec2 on an r5.large was chosen (this may be overkill but a t2.micro did not have enough memory, and initial attempts to deploy to lambda via sam cli failed due to lack of disk space on lambda). Elastic Beanstalk was chosen as the managed service to deploy the app to ec2 as it generates the auxiliary resources needed for the user.

To reproduce the deployment:

- install eb cli - this is `brew install awsebcli` on mac
- from `prediction` dir run `eb init -p python-3.7 mortgage-doc-predictor --region us-east-1` - initializes new elastic beanstalk project
- `eb config` - this opens elastic beanstalk configs in an editor - change `InstanceType` to `r5.large` and `WSGIPath` to `app.py`
- `eb create mortgage-doc-predictor-env` - begins creating of aws resources and deployment of app
- If you have already created your eb env and want to deploy code changes, use `eb deploy`
- `eb open` - once deployment is finished and ec2 is running and flask app is initialized, opens app in browser

App can be run locally with `flask run` and visited at `localhost:5000`

Currently running on AWS at http://mortgage-doc-predictor-env.6qukfteaus.us-east-1.elasticbeanstalk.com/ 

Testing the deployment:
- a basic test of the endpoint can be run by navigating to project root, running `. endpoint_curls.sh`, and manually inspecting results 

Note - in hindsight, this project could be improved by: 
1) training the model on ec2 during initialization of the webapp, removing the necessity of pickling objects and uploading the model.
2) improving results on classes with low numbers of examples - either by obtaining a more ideal dataset, or perhaps by generating fake data for those classes based on shuffled words from the real examples
3) deployment using cloudformation or SAM - initially I attempted to deploy this project to a lambda using SAM, but ran into space contraints due to the large libraries and the pickles needed to make predictions. That motivated my switch to an ec2 based deployment, but I have since learned about lambda layers which I suspect might solve the space issue, and would also conserve resources since the lambda would not need to be running constantly. Either way, using CloudFormation or Terraform would be preferable to the ElasticBeanstalk tool to further automate deployment.