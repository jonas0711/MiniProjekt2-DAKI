import cv2
import numpy as np
import os
import json

# --- Parametre ---
boards_dir = "KingDominoDataset/KingDominoDataset/Cropped and perspective corrected boards"  # Mappe med spilleplader
templates_dir = "KingDominoDataset/Crown_Templates"    # Mappe med krone-templates
threshold = 0.64                 # Match-tærskel
output_json_path = "crown_detection_results.json"      # Output JSON-fil

# --- Indlæs krone-templates ---
templates = []
for filename in os.listdir(templates_dir):
    if filename.endswith(('.jpg', '.png')):
        template_path = os.path.join(templates_dir, filename)
        template_img = cv2.imread(template_path)
        gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        templates.append((filename, gray_template))

# --- Gennemgå alle spilleplader ---
results = {}
for board_filename in os.listdir(boards_dir):
    if board_filename.endswith(('.jpg', '.png')):
        board_path = os.path.join(boards_dir, board_filename)
        board_img = cv2.imread(board_path)
        gray_board = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        board_results = {}
        for template_name, gray_template in templates:
            res = cv2.matchTemplate(gray_board, gray_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            h, w = gray_template.shape
            for pt in zip(*loc[::-1]):
                field_key = f"{pt[1] // h}_{pt[0] // w}"
                if field_key not in board_results:
                    board_results[field_key] = {"crowns": 0}
                board_results[field_key]["crowns"] += 1

        results[board_filename] = board_results

# --- Gem resultater i JSON-fil ---
with open(output_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Resultater gemt i {output_json_path}")