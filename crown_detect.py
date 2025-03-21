import cv2
import numpy as np
import os

# --- Parametre ---
board_path = "KingDominoDataset/KingDominoDataset/Cropped and perspective corrected boards/62.jpg"           # Billede af spillepladen
templates_dir = "KingDominoDataset/Crown_Templates"    # Mappe med krone-templates
threshold = 0.64                 # Match-tærskel

# --- Indlæs bræt og konverter til grå ---
board_img = cv2.imread(board_path)
gray_board = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

# --- Gennemgå alle templates og find kroner ---
for filename in os.listdir(templates_dir):
    if filename.endswith(('.jpg', '.png')):
        template_path = os.path.join(templates_dir, filename)
        template_img = cv2.imread(template_path)
        gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray_board, gray_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        h, w = gray_template.shape
        for pt in zip(*loc[::-1]):
            cv2.rectangle(board_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# --- Vis resultat ---
cv2.imshow('Kroner fundet', board_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- (Evt.) Gem resultat ---
# cv2.imwrite('output_med_kroner.jpg', board_img)
