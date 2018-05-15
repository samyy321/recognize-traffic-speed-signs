# recognize-traffic-speed-signs
Using my trained model to recognize traffic speed signs in image.

## How does it work?
The script restores a Tensorflow model that i trained for speed limit sign recognition (https://github.com/samyy321/training-speed-sign-classifier) and one of the 2 cascade classifiers trained for sign detection (thanks to https://github.com/putsi/tsaraisa).
Once the sign is detected, it creates a cropped image of the sign's placement and proceeds to recognition.
