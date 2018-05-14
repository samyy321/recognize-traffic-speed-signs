import tensorflow as tf
import cv2
import numpy as np
import operator

def restore_graph(sess, meta_path, ckpt_path):
	"""
	Restore tensorflow graph from latest checkpoint.
	"""
	saver = tf.train.import_meta_graph(meta_path)
	saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
	return tf.get_default_graph()

def classify_sign(sess, img, graph, classes, tensors_names):
	"""
	Run classification on image with restored model.

	The list of tensors names to restore is in the following order:
	[features, targets, conv_keep_prob, fc_keep_prob, op_to_restore].

	Returns a list of probabilities where prob[i] corresponds to classes[i].
	"""
	tensors = []
	for name in tensors_names:
		tensors.append(graph.get_tensor_by_name(name))

	images = []
	resized_img = cv2.resize(img, (tensors[0].shape[1], tensors[0].shape[2]))
	images.append(np.array(resized_img))
	x_batch = images
	targets = np.zeros((1, len(classes)))

	feed_dict = {tensors[0]: x_batch, tensors[1]: targets, tensors[2]: 1.0, tensors[3]: 1.0}
	return sess.run(tensors[4], feed_dict)[0]

def get_predicted_class(prob_list, classes):
	"""
	Returns the predicted class with his probability.
	"""
	idx, prob = max(enumerate(prob_list), key=operator.itemgetter(1))
	return classes[idx], prob
