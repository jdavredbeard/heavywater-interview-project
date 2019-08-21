from flask import Flask
from flask import request
import json
import os
import sys
from numpy import array
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf

sess = tf.Session()
set_session(sess)
model = load_model('mortgage_doc_mlp_model.h5')
graph = tf.get_default_graph()

pickle_in = open('pickles/label_encoder.pickle','rb')
label_encoder = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('pickles/vectorizer.pickle','rb')
vectorizer = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('pickles/selector.pickle','rb')
selector = pickle.load(pickle_in)
pickle_in.close()

application = Flask(__name__)

   
@application.route('/',methods=['GET'])
@application.route('/index',methods=['GET'])
def index():
    return 'hello heavywater!'

@application.route('/predict',methods=['GET'])
def predict():  
    args = request.args
    if args["words"]:
        words = args["words"]
        words = array([words])    
        words_vector = vectorizer.transform(words)
        words_vector = selector.transform(words_vector).astype('float32')
        global graph
        global sess
        with graph.as_default():
            set_session(sess)
            prediction = model.predict_classes(words_vector)
            prediction_class_encoded = prediction.tolist()[0]
            prediction_class_decoded = label_encoder.inverse_transform([prediction_class_encoded]).tolist()
            words = words.tolist()
    else:
        words = ""
        prediction_class_decoded = ['No words sent from which to make prediction']

    return {
        "statusCode": 200,
        "body": json.dumps({
            "words": words,
            "prediction": prediction_class_decoded
        }),
        "headers": {
            "Access-Control-Allow-Origin": "http://jdavredbeard-website-bucket.s3-website-us-east-1.amazonaws.com"
        }
    }


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
# print(handler({"queryStringParameters":
#     {"words":"e04a09c87692 d6b72e591b91 5d066f0246f1 ed41171e2e6d 59260a2781dc ec56ff31bb7a 1cf70e99f986 7d7400d32c11 fbe7c05e32d5 6b0cb5728b14 54709b24b45f 25c57acdf805 8bd6e6f02cbc 31cbd98f4b3c f7548baf29d4 bd0972f16400 1b43925e3c28 b2c878a75d7e 59260a2781dc cf4fc632eed2 25c57acdf805 b4221b1edff9 de9738ee8b24 135307dba198 19e9f3592995 1cf70e99f986 266dc1fd820c b73e657498f2 f1ec22325b37 1fa87d60c46c e4a319284bf9 6b343f522f78 60fb2adbbb87 37428698b32e d03283541bce 59f0408bc81b 9ccf259ca087 54709b24b45f 8bd6e6f02cbc 63198bea516d 7991590bf0b6 2575240863a4 b5ed9af384f4 87b8193a0183 3e5199ae28ae 094e2de7e1cd 422068f04236 25c57acdf805 179dce4734b4 d38820625542 7cd2e94152fb f1413affa34b 288ccf089872 3eba94383fac e1af6790122f f36858486ddc ec3406979928 cff184767152 da9ad7407226 0c222c6660f2 586242498a88 d1c3631d621b 323c2f20cf45 25c57acdf805 2190972b219c f2b0e028fe2c da61efdd2b77 9cdf4a63deb0 6a95ce91efbd 432bd6e0c08f 6b343f522f78 8c05761f94b8 b7ab56536ec4 afb1e3806fc1 c0d455ef7403 cafaf222091d 7cd2e94152fb 9974e72c1e3d 586242498a88 ffe8decfd82e d38820625542 4e5019f629a9 7cd2e94152fb 179dce4734b4 586242498a88 86f5abd0f322 2de82c64620a da9ad7407226 d38820625542 133d46f7ed38 da61efdd2b77 af671fbeb212 d4bec079d88a 7d9e333a86da cafaf222091d 7cd2e94152fb 7c0d350ba703 d38820625542 7cd2e94152fb bce2a87e8f26 4e5019f629a9 c337a85b8ef9 e04a09c87692 4ad52689d690 6b304aabdcee ce1f034abb5d 28bb12cc84c2 9f11111004ec eeb86a6a04e4 f9b20c280980 93a5aefea103 083656409314 eaa9c2339ac6 8b6c5bb157a9 fc9ae3c91204 7ec02e30a5b3 44ff8f74af45 33fb7d5700b3 7d9e333a86da 3e24b036118a 479e6e2b5afc 3cd4957ae1e5 52102c70348d 36e7aa72ffe1 c337a85b8ef9 cafaf222091d b9699ce57810 e7cd8a78bd1c 1015893e384a 586242498a88 b208ae1e8232 0992140a4618 30025054f25c 1015893e384a 8b6c5bb157a9 fc9ae3c91204 7ec02e30a5b3 44ff8f74af45 33fb7d5700b3 d19e3740b9f4 fb8ac3af8dce 1f72b4630c4a 6bf9c0cb01b4 e94953618947 c337a85b8ef9 083656409314 eaa9c2339ac6 b2bf1b3a363b 21300e217202 2979f8b5c8c4 557ec6c63cf9 b9e341c696e7 4e43b72d46c0 816a114f1b9a 02db3d3134b6 e76a3d82e892 8f75273e5510 43565b1afa44 1807f8910862 2ed97f462806 c8f5ad40a683 f0fd45d01d0a dec88250479b 0302af775b89 cafaf222091d 6c14785745cb 8b6c5bb157a9 e9849b397690 c33578d25a0d 46142e2508e4 036087ac04f9 b136f6349cf3 dee46d35626e d38820625542 8f75273e5510 43565b1afa44 bad6ff5dd7bc 420691c4fc64 2de82c64620a b9699ce57810 cafaf222091d 4e9eb063e763 dee46d35626e 232d63889bcf 056314258a60 d1c45a16f923 d38820625542 8f75273e5510 43565b1afa44 4e5019f629a9 dec88250479b cafaf222091d cafaf222091d f52871480463 6d1fb90988cf 7cd2e94152fb 8b6c5bb157a9 d4e08985be1b e9849b397690 816a114f1b9a 4e43b72d46c0 b9e341c696e7 02db3d3134b6 9ad186d42f69 036087ac04f9 cafaf222091d b136f6349cf3 c33578d25a0d 46142e2508e4 2979f8b5c8c4 21300e217202 b2bf1b3a363b c8f5ad40a683 557ec6c63cf9 586242498a88 cde4f1b2a877 ec56ff31bb7a d38820625542 7cd2e94152fb 25c57acdf805 d572dc50fdd4 133d46f7ed38 4e5019f629a9 8f67fb7907c9 a86f2ba617ec c75cfd6fc902 a6cada1d2f54 f898dba78476 da61efdd2b77 586242498a88 a98f06677951 8b6c5bb157a9 02db3d3134b6 78ef61730b15 7cd2e94152fb d38820625542 586242498a88 7cd2e94152fb f52871480463 b136f6349cf3 036087ac04f9 46142e2508e4 c33578d25a0d e9849b397690 d38820625542 dee46d35626e 2979f8b5c8c4 21300e217202 b2bf1b3a363b 557ec6c63cf9 036087ac04f9 75440bb763a2 816a114f1b9a 4e43b72d46c0 b9e341c696e7 a31962fbd5f3 4357c81e10c1 93790ade6682 61b7e0f00ffe 8f75273e5510 43565b1afa44 9cdf4a63deb0 6ca2dd348663 2ef7c27a5df4 b136f6349cf3 0f88ca127938 57962002319d e0e86b8f64e1 8b6c5bb157a9 3102eeb23202 b274cd8dd187 c3d9d343468b 7464df526d87 5079f87b7bd6 cbd6e2e7a804 9bfc4c4973a6 8f75273e5510 43565b1afa44 1807f8910862 b77b531ae299 743b314e5665 d38820625542 7cd2e94152fb 3cd4957ae1e5 8754554be158 7cd2e94152fb dee46d35626e 010bdb69ff0a c337a85b8ef9 4ad52689d690 ce1f034abb5d 28bb12cc84c2 9f11111004ec f9b20c280980 d38820625542 6a95ce91efbd 816a114f1b9a 4e43b72d46c0 b9e341c696e7 62a9fc81b3e8 7cd2e94152fb d38820625542 586242498a88 54709b24b45f 1ab34730c1e0 2b2acdf4a3ed 65fa97636e09 1b398b5b9d07 8db69e29b4cf 1c303d15eb65 76957326a333 6fbb21936a35 e8d2105b441f 89aea207f9e3 6121e3c3fa4d 3cd4957ae1e5 918d14133622 421e52f8278f be20e2fc1f33 a86f2ba617ec 0562c756a2f2 431392fb12c7 1b6d0614f2c7 b208ae1e8232 4a0084089b01 4b47049a7fd1 572b3238a2c4 5c02c2aaa67b 628863dfcdcd 6c8642055a4e f0fd45d01d0a 572b3238a2c4 da2e1b058e4b ffca4a0468c8 eeb86a6a04e4 572b3238a2c4 fe64d3cdfe5b da2e1b058e4b 1c303d15eb65 2c33750c1d59 ed1e3242ee34 2a6fbe672600 69056c5109b2 572b3238a2c4 9da09997e84d 97b6014f9e50 628863dfcdcd 93790ade6682 61b7e0f00ffe 4357c81e10c1 4e5019f629a9 a31962fbd5f3 26f7353edc2e 9b002bb8ea70 0056a542a568 d38820625542 9f11111004ec f9b20c280980 c337a85b8ef9 4ad52689d690 5ee06767bc0f 75440bb763a2 1ab34730c1e0 036087ac04f9 6b343f522f78 b136f6349cf3 2ef7c27a5df4 6ca2dd348663 a1e5904d04f5 d38820625542 3d50cd6a0f95 5ee06767bc0f 2b2acdf4a3ed 7d41ca882f26 b9f6c678fbb0 af3bfda79f9d 6ce6cc5a3203 798fe9915030 0f408aaf2b1c cfefa92a643c 1015893e384a ce1f034abb5d 28bb12cc84c2 67f9c02008c6 586242498a88 586242498a88 6685a4a0c6a4 2094595088dd ad062bd785fb 8b6c5bb157a9 1015893e384a 8288c343e374 5e99d31d8fa4 c33578d25a0d b9f6c678fbb0 3d478e13c895 af3bfda79f9d 2094595088dd 9b002bb8ea70 036087ac04f9 0056a542a568 2bcce4e05d9d f700adeb40f0 f700adeb40f0 392ef61e1c96 392ef61e1c96 60304c03969f f0666bdbc8a5 6ce6cc5a3203 586242498a88 6a01047db3ab 2685f0879380 2a6fbe672600 e828da93d525 cb7631b88e51 580a08f5c8b9 45e16cbae38f 1015893e384a 094453b4e4ae 586242498a88 b73e657498f2 6ca2dd348663 d38820625542 9b002bb8ea70 0056a542a568 8f75273e5510 93790ade6682 61b7e0f00ffe 4357c81e10c1 a31962fbd5f3 75440bb763a2 036087ac04f9 b136f6349cf3 2ef7c27a5df4 a1e5904d04f5 3d50cd6a0f95 2b2acdf4a3ed 6b304aabdcee 6b304aabdcee 6b304aabdcee 30576f6a6e96 6b304aabdcee f8931fcd9bd6 f8931fcd9bd6 f8931fcd9bd6 4d9284df2a3f 6b304aabdcee c1b3166ecf3b 6b304aabdcee 6b304aabdcee b9a670c30df5 41a0ee0cf31a c87d29676bd1 4faebb8051b2 22d6fd31d92b e3d61318ef03 036087ac04f9 2bcce4e05d9d 010bdb69ff0a 44d3870bca21 4e5019f629a9 586242498a88 6ca2dd348663 b208ae1e8232 04503bc22789"}},{}))
