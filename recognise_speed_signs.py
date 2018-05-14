import argparse
import tensorflow as tf
import cv2
import os

import classify

CLASSIFIER_PATH = "./classifiers/haarCascade.xml"
GRAPH_META = './speed-sign-recognizer-model/speed-sign-recognizer.meta'
GRAPH_CKPT_DIR = './speed-sign-recognizer-model/'

def crop_sign(img, x, y, width, height, padding):
	"""
	Crops the detected sign starting with a padding
	and reduces it if the cropped image dexceeds the original image.
	"""
	i = padding
	while True:
		cropped = img[y - i:y + height + i, x - i:x + width + i]
		if cropped.any() or i == 0:
			return cropped
		i -= 1
	return None

def get_args():
	parser = argparse.ArgumentParser(
		description='Detect and recognize traffic speed signs.')
	parser.add_argument(
		"images",
		help="Paths of the images you want to process.",
		nargs='+')
	parser.add_argument(
		"--classifier-path",
		help="Path of the OpenCV cascade classifier you want to use.",
		default=CLASSIFIER_PATH)
	parser.add_argument(
		"--graph-meta-path",
		help="Path of the Tensorflow model meta.",
		default=GRAPH_META)
	parser.add_argument(
		"--graph-ckpt-path",
		help="Directory path of the Tensorflow model checkpoint.",
		default=GRAPH_CKPT_DIR)
	return parser.parse_args()

def main():
	args = get_args()

	sess = tf.Session()
	graph = classify.restore_graph(sess, args.graph_meta_path, args.graph_ckpt_path)

	classifier = cv2.CascadeClassifier(args.classifier_path)
	for img_path in args.images:
		print('Processing %s...' % img_path)
		img = cv2.imread(img_path)
		if img is None:
			print("Unable to read image from path.")
			continue
		signs = classifier.detectMultiScale(img, 1.1, 5, 0, (10, 10), (200, 200))
		if len(signs) == 0:
			print("No speed signs detected.")
			continue
		clone = img.copy()
		for sign in signs:
			x, y, width, height = [ i for i in sign ]
			cropped = crop_sign(img, x, y, width, height, 15)
			classes = ['30', '50', '60', '70', '80']
			tensors_names = ['features:0', 'targets:0', 'conv_dropout:0', 'fc_dropout:0', 'y_pred:0']
			probabilities = classify.classify_sign(sess, cropped, graph, classes, tensors_names)
			sign_class, prob = classify.get_predicted_class(probabilities, classes)
			cv2.rectangle(clone, (x, y), (x + width, y + height), (0, 255, 0), 2)
			cv2.putText(clone, str(sign_class), (x - 30, y + height), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
		cv2.imwrite(os.path.splitext(img_path)[0] + "_detections.jpg", clone)
		print('Done.')

if __name__ == '__main__':
	main()
