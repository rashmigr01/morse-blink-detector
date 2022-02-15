# Morse Code Blink Detector

Detects and translates blinks in morse code.



## Run Locally

Clone the project:

```bash
  git clone https://github.com/rashmigr01/morse-blink-detector.git
```



## Requirements

Install a pre-trained facial landmark predictor, shape_predictor_68_face_landmarks.dat by decompressing the .bz2 file available on the [dlib site](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
<br /> Also available [here](https://github.com/davisking/dlib-models/blob/daf943f7819a3dda8aec4276754ef918dc26491f/shape_predictor_68_face_landmarks.dat.bz2).
<br />Packages OpenCV, argparse and imutils.

    
## Run using the command line

```javascript
python blinks.py -o shape_predictor_68_face_landmarks.dat
```
