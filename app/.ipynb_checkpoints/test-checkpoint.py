import cnn_lstm_ as c1
strin="Great phone, I was concerned about buying used however it has turned out a very good decision."

print("Loading saved model...")

print("This is the ans"+str(c1.predict_response(strin)))

#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))