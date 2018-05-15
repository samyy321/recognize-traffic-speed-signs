# recognize-traffic-speed-signs
Using my trained model to recognize traffic speed signs in image.

## How does it work?
The script restores a Tensorflow model that i trained for speed limit sign recognition (https://github.com/samyy321/training-speed-sign-classifier) and one of the 2 cascade classifiers trained for sign detection (thanks to https://github.com/putsi/tsaraisa).
Once the sign is detected, it creates a cropped image of the sign's placement and proceeds to recognition.

## Usage
```
usage: recognise_speed_signs.py [-h] [--classifier-path CLASSIFIER_PATH]
                                [--graph-meta-path GRAPH_META_PATH]
                                [--graph-ckpt-path GRAPH_CKPT_PATH]
                                images [images ...]

Detect and recognize traffic speed signs.

positional arguments:
  images                Paths of the images you want to process.

optional arguments:
  -h, --help            show this help message and exit
  --classifier-path CLASSIFIER_PATH
                        Path of the OpenCV cascade classifier you want to use.
  --graph-meta-path GRAPH_META_PATH
                        Path of the Tensorflow model meta.
  --graph-ckpt-path GRAPH_CKPT_PATH
                        Directory path of the Tensorflow model checkpoint.
```

The script uses the HAAR cascade classifier present in this repository by default
and a Tensorflow model that is not present in this repository you can train your
own by using the same architecture that i used in https://github.com/samyy321/training-speed-sign-classifier.
