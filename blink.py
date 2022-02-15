from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import keyboard

ethresh = 0.20
eframes = 3
eclosed = 10
pframes = 20
wframes = 20
bframes = 30

AlphaToMorse = {'a': ".-", 'b': "-...", 'c': "-.-.", 'd': "-..", 'e': ".",
                'f': "..-.", 'g': "--.", 'h': "....", 'i': "..", 'j': ".---", 'k': "-.-",
                'l': ".-..", 'm': "--", 'n': "-.", 'o': "---", 'p': ".--.", 'q': "--.-",
                'r': ".-.", 's': "...", 't': "-", 'u': "..-", 'v': "...-", 'w': ".--",
                'x': "-..-", 'y': "-.--", 'z': "--..",
                '1': ".----", '2': "..---", '3': "...--", '4': "....-", '5': ".....",
                '6': "-....", '7': "--...", '8': "---..", '9': "----.", '0': "-----",
                ' ': "¦", '.': ".-.-.-", ',': "--..--", '?': "..--..", "'": ".----.",
                '@': ".--.-.", '-': "-....-", '"': ".-..-.", ':': "---...", ';': "---...",
                '=': "-...-", '!': "-.-.--", '/': "-..-.", '(': "-.--.", ')': "-.--.-"}

MORSE = {value:key for key,value in AlphaToMorse.items()}

def from_morse(s):
    result = ""
    for i in s.split("/"):
        if i in MORSE:
            result += MORSE.get(i)
        else:
            if (i != ""):
                print(i + " could not be translated.")
    return result

def main():
	arg_par = argparse.ArgumentParser()
	arg_par.add_argument("-o", "--shape-predictor", required=True)
	args = vars(arg_par.parse_args())

	(vs, detector, predictor, lStart, lEnd, rStart,
			rEnd) = setup_detector_video(args)
	total_morse = loop_camera(vs, detector, predictor, lStart, 
			lEnd, rStart, rEnd)
	cleanup(vs)
	print_results(total_morse)

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	eye_ar = (A + B) / (2.0 * C)
	return eye_ar

def setup_detector_video(args):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	print("Type ';' or close eyes for {} frames to exit.".format(bframes))
	vs = VideoStream(src=0).start()
	return vs, detector, predictor, lStart, lEnd, rStart, rEnd

def loop_camera(vs, detector, predictor, lStart, lEnd, rStart, rEnd):
	counter = 0
	break_counter = 0
	eyes_open = 0
	eyes_closed = False
	pause = False
	ispaused = False

	total_morse = ""
	morse_word = ""
	morse_char = ""

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		rects = detector(gray, 0)

		for rect in rects:
			
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			left_eye_ar = eye_aspect_ratio(leftEye)
			right_eye_ar = eye_aspect_ratio(rightEye)
			
			eye_ar = (left_eye_ar + right_eye_ar) / 2.0

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			if eye_ar < ethresh:
				counter += 1
				break_counter += 1
				if counter >= eframes:
					eyes_closed = True
				
				if not ispaused:
					morse_char = ""
				 
				if (break_counter >= bframes):
					break
			else:
				if (break_counter < bframes):
					break_counter = 0
				eyes_open += 1
				
				if counter >= eclosed:
					morse_word += "-"
					total_morse += "-"
					morse_char += "-"
					
					counter = 0
					eyes_closed = False
					ispaused = True
					eyes_open = 0
				
				elif eyes_closed:
					morse_word += "."
					total_morse += "."
					morse_char += "."
					counter = 1
					eyes_closed = False
					ispaused = True
					eyes_open = 0
				
				elif ispaused and (eyes_open >= pframes):
					morse_word += "/"
					total_morse += "/"
					morse_char = "/"
					ispaused = False
					pause = True
					eyes_closed = False
					eyes_open = 0
					keyboard.write(from_morse(morse_word))
					morse_word = ""
				
				elif (pause and eyes_open >= wframes):
					total_morse += "¦/"
					morse_char = ""
					pause = False
					eyes_closed = False
					eyes_open = 0
					keyboard.write(from_morse("¦/"))

			cv2.putText(frame, "EAR: {:.2f}".format(eye_ar), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "{}".format(morse_char), (100, 200),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

			print("\033[K", "morse_word: {}".format(morse_word), end="\r")

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord(";") or (break_counter >= bframes):
			keyboard.write(from_morse(morse_word))
			break
	return total_morse

def cleanup(vs):
	cv2.destroyAllWindows()
	vs.stop()

def print_results(total_morse):
	print("Morse Code: ", total_morse.replace("¦", " "))
	print("Translated: ", from_morse(total_morse))

if __name__ == "__main__":
	main()