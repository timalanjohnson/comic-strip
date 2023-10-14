import cv2
import numpy as np
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import words

print('Load the image')
image = cv2.imread('./images/input/page.jpg')

print('Apply edge detection')
edges = cv2.Canny(image, 50, 150, apertureSize=3)

print('Find contours')
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print('Find the longest contour')
max_contour = max(contours, key=cv2.contourArea)

print('Approximate the contour to a polygon with 4 points')
epsilon = 0.03 * cv2.arcLength(max_contour, True)
approx = cv2.approxPolyDP(max_contour, epsilon, True)

print('Get the bounding rectangle of the contour')
x, y, w, h = cv2.boundingRect(approx)

print('Calculate the aspect ratio of the bounding rectangle')
aspect_ratio = float(w) / h

print('Create a target rectangle with the same aspect ratio')
target_width = 4096
target_height = int(target_width / aspect_ratio)

print('Define the corners of the target rectangle')
target_corners = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)

print('Warp the image to fit the target rectangle')
matrix = cv2.getPerspectiveTransform(approx.reshape(4, 2).astype(np.float32), target_corners)
cropped = cv2.warpPerspective(image, matrix, (target_width, target_height))

print('Flip the image')
flipped = cv2.flip(cropped, 1)

result = flipped

print('Extract text from image')
extracted_text = pytesseract.image_to_string(result)

print('Download punkt')
nltk.download('punkt')

print('Download words')
nltk.download('words')

print('Tokenize the OCR result')
words_in_text = nltk.word_tokenize(extracted_text)

# Filter out non-English words
english_words = set(words.words())
filtered_words = [word for word in words_in_text if word.lower() in english_words]

# Reconstruct the filtered text into sentences
filtered_text = " ".join(filtered_words)

# Optionally, display the filtered text
print("Filtered Text:")
print(filtered_text)

cv2.imwrite('result.jpg', result)

pil_image = Image.open('result.jpg')

pil_image.info['Description'] = filtered_text

pil_image.save('./images/output/result.jpg')

print('Display or save the result')
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()