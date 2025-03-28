import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm  # Brug gammel import
import cv2
import pickle
from collections import deque

from old.crown_detector import detect_crowns_positions  # Brug den nye funktion

# Standardstier
MODEL_FILE = "kingdomino_terrain_model.pkl"
DEFAULT_IMAGE_PATH = "KingDominoDataset/KingDominoDataset/Cropped and perspective corrected boards/7.jpg"
DEFAULT_OUTPUT_PATH = "kingdomino_result.png"

# Vi beholder TerrainClassifier og featurefunktionerne som før
class TerrainClassifier:
    def __init__(self, lda=None, knn=None, terrain_classes=None):
        self.lda = lda
        self.knn = knn
        self.terrain_classes = terrain_classes
        self.is_fitted = (lda is not None and knn is not None)
    
    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.neighbors import KNeighborsClassifier
        n_components = min(len(np.unique(y)), X.shape[1]) - 1
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = self.lda.fit_transform(X, y)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(X_lda, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Modellen er ikke trænet endnu.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_lda = self.lda.transform(X)
        return self.knn.predict(X_lda)
    
    def predict_terrain(self, X):
        if self.terrain_classes is None:
            raise ValueError("Terrain classes mapping er ikke tilgængelig.")
        y_pred = self.predict(X)
        terrain_names = {v: k for k, v in self.terrain_classes.items()}
        return [terrain_names[label] for label in y_pred]

def extract_hsv_histogram(image, bins=32):
    try:
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    except:
        return np.zeros(bins * 3)
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return np.concatenate([hist_h, hist_s, hist_v])

def extract_texture_histogram(image, bins=9):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except:
        return np.zeros(bins)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi) % 180
    hist = np.zeros(bins)
    bin_width = 180 / bins
    for i in range(bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        mask = ((direction >= bin_start) & (direction < bin_end))
        hist[i] = np.sum(magnitude[mask])
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    return hist

def extract_features(image):
    hsv_hist = extract_hsv_histogram(image)
    texture_hist = extract_texture_histogram(image)
    return np.concatenate([hsv_hist, texture_hist])

def load_model(file_path=MODEL_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Modelfilen {file_path} findes ikke.")
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model indlæst fra: {file_path}")
    return model

def load_board_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Billedfilen {image_path} findes ikke.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Kunne ikke indlæse billedet fra {image_path}.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def divide_board_into_tiles(image, grid_size=5):
    height, width, _ = image.shape
    tile_height = height // grid_size
    tile_width = width // grid_size
    tiles = []
    # Vi gemmer også top-left koordinater for hver tile
    tile_coords = []
    for row in range(grid_size):
        tile_row = []
        coord_row = []
        for col in range(grid_size):
            y_start = row * tile_height
            x_start = col * tile_width
            tile = image[y_start:y_start + tile_height, x_start:x_start + tile_width]
            tile_row.append(tile)
            coord_row.append((x_start, y_start, tile_width, tile_height))
        tiles.append(tile_row)
        tile_coords.append(coord_row)
    return tiles, tile_coords

def classify_tiles_with_crowns(tiles, model):
    grid_size = len(tiles)
    terrain_results = np.empty((grid_size, grid_size), dtype=object)
    crown_results = np.empty((grid_size, grid_size), dtype=int)
    crown_positions = {}  # nøgler: (row, col), værdi: liste af centroids (lokal i tile)
    
    all_features = []
    tile_positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            features = extract_features(tiles[row][col])
            all_features.append(features)
            tile_positions.append((row, col))
    
    all_features = np.array(all_features)
    terrain_types = model.predict_terrain(all_features)
    
    for i, (row, col) in enumerate(tile_positions):
        # Hårdkod midterfeltet (2,2) til 'Home'
        if (row, col) == (2, 2):
            terr = 'Home'
        else:
            terr = terrain_types[i]
        terrain_results[row, col] = terr
        
        if terr in ['Home', 'Unknown']:
            crown_results[row, col] = 0
            crown_positions[(row, col)] = []
            continue
        
        tile_img = tiles[row][col]
        count, centroids = detect_crowns_positions(tile_img, terr)
        crown_results[row, col] = count
        crown_positions[(row, col)] = centroids
    
    return terrain_results, crown_results, crown_positions

def identify_connected_territories(terrain_results, crown_results):
    grid_size = len(terrain_results)
    visited = np.zeros((grid_size, grid_size), dtype=bool)
    territories = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for row in range(grid_size):
        for col in range(grid_size):
            if visited[row, col] or terrain_results[row, col] in ['Home', 'Unknown']:
                continue
            current_terrain = terrain_results[row, col]
            territory_tiles = []
            total_crowns = 0
            queue = deque([(row, col)])
            visited[row, col] = True
            while queue:
                r, c = queue.popleft()
                territory_tiles.append((r, c))
                total_crowns += crown_results[r, c]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < grid_size and 0 <= nc < grid_size and
                        not visited[nr, nc] and
                        terrain_results[nr, nc] == current_terrain):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            score = len(territory_tiles) * total_crowns if total_crowns > 0 else 0
            territories.append({
                'terrain': current_terrain,
                'tiles': territory_tiles,
                'crowns': total_crowns,
                'score': score
            })
    return territories

def score_board(terrain_results, crown_results):
    territories = identify_connected_territories(terrain_results, crown_results)
    grid_size = len(terrain_results)
    has_harmony = all(terrain_results[r, c] != 'Unknown' for r in range(grid_size) for c in range(grid_size))
    harmony_bonus = 5 if has_harmony else 0
    total_score = sum(t['score'] for t in territories) + harmony_bonus
    return {'territories': territories, 'harmony_bonus': harmony_bonus, 'total_score': total_score}

def visualize_scored_board(original_image, tiles, tile_coords, terrain_results, crown_results, crown_positions, score_result, output_path=None):
    grid_size = len(tiles)
    terrain_colors = {
        'Field': 'gold',
        'Forest': 'darkgreen',
        'Lake': 'lightblue',
        'Mine': 'saddlebrown',
        'Swamp': 'olive',
        'Grassland': 'limegreen',
        'Home': 'gray',
        'Unknown': 'pink'
    }
    
    # Vis det originale board med overlejrede stjerner på de fundne kroner
    board_vis = original_image.copy()
    for row in range(grid_size):
        for col in range(grid_size):
            x, y, w, h = tile_coords[row][col]
            centroids = crown_positions.get((row, col), [])
            # Tegn stjerne for hver fundet krone (global position)
            for (cx, cy) in centroids:
                global_x = x + cx
                global_y = y + cy
                # Brug plt.text senere, men her kan vi også tegne med cv2
                cv2.putText(board_vis, "*", (global_x, global_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Visualiser med matplotlib
    plt.figure(figsize=(15, 15))
    
    # (1) Original plade med stjerner
    plt.subplot(2, 2, 1)
    plt.imshow(board_vis)
    plt.title("Original plade med fundne kroner (stjerner)")
    plt.axis('off')
    
    # (2) Terræn + kroner (som før)
    plt.subplot(2, 2, 2)
    terrain_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            terr = terrain_results[r, c]
            color = terrain_colors.get(terr, 'white')
            rgb = mcolors.to_rgb(color)
            terrain_img[r, c] = [int(x*255) for x in rgb]
    plt.imshow(terrain_img)
    for r in range(grid_size):
        for c in range(grid_size):
            terr = terrain_results[r, c]
            cr = crown_results[r, c]
            crown_text = '*' * cr if cr > 0 else ''
            plt.text(c, r, f"{terr}\n{crown_text}", ha="center", va="center",
                     color="white", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    plt.title("Terrænklassifikation med kroner")
    plt.axis('off')
    
    # (3) Territorier med scores (som før)
    plt.subplot(2, 2, 3)
    territory_mask = np.zeros((grid_size, grid_size), dtype=int)
    for i, territory in enumerate(score_result['territories']):
        for (r, c) in territory['tiles']:
            territory_mask[r, c] = i + 1
    territory_cmap = cm.get_cmap('tab20', len(score_result['territories']) + 1)
    territory_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            tid = territory_mask[r, c]
            if tid > 0:
                rgb = territory_cmap(tid)[:3]
                territory_img[r, c] = [int(x*255) for x in rgb]
            else:
                terr = terrain_results[r, c]
                color = terrain_colors.get(terr, 'white')
                rgb = mcolors.to_rgb(color)
                territory_img[r, c] = [int(x*255) for x in rgb]
    plt.imshow(territory_img)
    territory_scores = {}
    for territory in score_result['territories']:
        r_avg = sum(t[0] for t in territory['tiles']) / len(territory['tiles'])
        c_avg = sum(t[1] for t in territory['tiles']) / len(territory['tiles'])
        territory_scores[(r_avg, c_avg)] = territory['score']
    for (rr, cc), sc in territory_scores.items():
        plt.text(cc, rr, f"{sc}", ha="center", va="center",
                 color="white", fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="circle,pad=0.3", fc="black", alpha=0.7))
    plt.title("Territorier med scores")
    plt.axis('off')
    
    # (4) Score detaljer
    plt.subplot(2, 2, 4)
    plt.axis('off')
    sorted_territories = sorted(score_result['territories'], key=lambda t: t['score'], reverse=True)
    score_text = "KINGDOMINO SCORE\n\n"
    for i, t in enumerate(sorted_territories):
        terr = t['terrain']
        tile_count = len(t['tiles'])
        cr = t['crowns']
        sc = t['score']
        score_text += f"Territorium {i+1} ({terr}):\n  {tile_count} felter × {cr} kroner = {sc} points\n\n"
    score_text += f"Harmony Bonus: {score_result['harmony_bonus']} points\n"
    score_text += f"Total Score: {score_result['total_score']} points"
    plt.text(0.5, 0.5, score_text, ha="center", va="center", fontsize=12,
             bbox=dict(boxstyle="round,pad=1", fc="white", alpha=0.9))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Score visualisering gemt til {output_path}")
    else:
        plt.show()

def full_pipeline(image_path=DEFAULT_IMAGE_PATH, model_path=MODEL_FILE, output_path=DEFAULT_OUTPUT_PATH, grid_size=5):
    try:
        print(f"Indlæser model fra {model_path}...")
        model = load_model(model_path)
        print(f"Indlæser pladebillede fra {image_path}...")
        board_image = load_board_image(image_path)
        print(f"Opdeler pladen i {grid_size}x{grid_size} tiles...")
        tiles, tile_coords = divide_board_into_tiles(board_image, grid_size)
        print("Klassificerer tiles og detekterer kroner...")
        terrain_results, crown_results, crown_positions = classify_tiles_with_crowns(tiles, model)
        print("Beregner score...")
        score_result = score_board(terrain_results, crown_results)
        print("Visualiserer resultater med stjerner for fundne kroner...")
        visualize_scored_board(board_image, tiles, tile_coords, terrain_results, crown_results, crown_positions, score_result, output_path)
        print("\nKlassifikationsresultat med kroner:")
        for r in range(grid_size):
            row_list = []
            for c in range(grid_size):
                terr = terrain_results[r, c]
                cr = crown_results[r, c]
                row_list.append(f"{terr}({cr}*)")
            print(" ".join(row_list))
        print(f"\nTotal score: {score_result['total_score']} points")
        print(f"\nFuldført! Resultat gemt til {output_path}")
        return score_result
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fejl: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Kør Kingdomino analyse pipeline')
    parser.add_argument('--image', default=DEFAULT_IMAGE_PATH, help='Sti til billedet af spillepladen')
    parser.add_argument('--model', default=MODEL_FILE, help='Sti til den gemte model')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_PATH, help='Sti til at gemme visualisering')
    parser.add_argument('--grid_size', type=int, default=5, help='Størrelse af grid (standard: 5)')
    args = parser.parse_args()
    full_pipeline(args.image, args.model, args.output, args.grid_size)

if __name__ == "__main__":
    main()
