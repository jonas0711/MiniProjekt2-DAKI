import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import mcolors
from old.crown_detector import detect_crowns
from model import load_model, extract_features, TerrainClassifier

MODEL_FILE = "kingdomino_terrain_model.pkl"
DEFAULT_IMAGE_PATH = r"KingDominoDataset\KingDominoDataset\Cropped and perspective corrected boards\1.jpg"

def load_board_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Billedfilen {image_path} findes ikke.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Kunne ikke indlæse billedet fra {image_path}.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def divide_board_into_tiles(image, grid_size=5):
    height, width, _ = image.shape
    tile_height = height // grid_size
    tile_width = width // grid_size
    tiles = []
    for row in range(grid_size):
        tile_row = []
        for col in range(grid_size):
            y_start = row * tile_height
            x_start = col * tile_width
            tile = image[y_start:y_start + tile_height, x_start:x_start + tile_width]
            tile_row.append(tile)
        tiles.append(tile_row)
    return tiles

def classify_tiles_with_crowns(tiles, model):
    grid_size = len(tiles)
    terrain_results = np.empty((grid_size, grid_size), dtype=object)
    crown_results = np.empty((grid_size, grid_size), dtype=int)
    all_features = []
    tile_positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            tile_features = extract_features(tiles[row][col])
            if not isinstance(tile_features, np.ndarray):
                tile_features = np.array(tile_features)
            all_features.append(tile_features)
            tile_positions.append((row, col))
    all_features = np.array(all_features)
    terrain_types = model.predict_terrain(all_features)
    for i, (row, col) in enumerate(tile_positions):
        terrain_type = terrain_types[i]
        terrain_results[row, col] = terrain_type
        if terrain_type in ['Home', 'Unknown']:
            crown_results[row, col] = 0
            continue
        crown_count = detect_crowns(tiles[row][col], terrain_type)
        crown_results[row, col] = crown_count
    return terrain_results, crown_results

def visualize_classification_with_crowns(original_image, tiles, terrain_results, crown_results, output_path=None):
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
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original plade")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    classification_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    for row in range(grid_size):
        for col in range(grid_size):
            terrain = terrain_results[row, col]
            color = terrain_colors.get(terrain, 'white')
            if color == 'gold':
                rgb = [255, 215, 0]
            elif color == 'darkgreen':
                rgb = [0, 100, 0]
            elif color == 'lightblue':
                rgb = [173, 216, 230]
            elif color == 'saddlebrown':
                rgb = [139, 69, 19]
            elif color == 'olive':
                rgb = [128, 128, 0]
            elif color == 'limegreen':
                rgb = [50, 205, 50]
            elif color == 'gray':
                rgb = [128, 128, 128]
            elif color == 'pink':
                rgb = [255, 192, 203]
            else:
                rgb = [255, 255, 255]
            classification_img[row, col] = rgb
    plt.imshow(classification_img)
    for row in range(grid_size):
        for col in range(grid_size):
            terrain = terrain_results[row, col]
            crowns = crown_results[row, col]
            crown_text = '⭐' * crowns if crowns > 0 else ''
            label_text = f"{terrain}\n{crown_text}"
            plt.text(col, row, label_text, ha="center", va="center", 
                     color="white", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5))
    plt.title("Klassifikation med kroner")
    plt.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisering gemt til {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Klassificer King Domino-plade med kroner')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE_PATH, help='Sti til billedet af spillepladen')
    parser.add_argument('--model', type=str, default=MODEL_FILE, help='Sti til den gemte model')
    parser.add_argument('--output', type=str, default='classification_with_crowns.png', help='Sti til at gemme visualisering')
    parser.add_argument('--grid_size', type=int, default=5, help='Størrelse af grid (standard: 5)')
    args = parser.parse_args()
    try:
        print(f"Indlæser model fra {args.model}...")
        model = load_model(args.model)
        print(f"Indlæser pladebillede fra {args.image}...")
        board_image = load_board_image(args.image)
        print(f"Opdeler pladen i {args.grid_size}x{args.grid_size} tiles...")
        tiles = divide_board_into_tiles(board_image, args.grid_size)
        print("Klassificerer tiles og detekterer kroner...")
        terrain_results, crown_results = classify_tiles_with_crowns(tiles, model)
        print("Visualiserer resultater...")
        visualize_classification_with_crowns(board_image, tiles, terrain_results, crown_results, args.output)
        print("\nKlassifikationsresultat med kroner:")
        for row in range(len(terrain_results)):
            print(" ".join([f"{terrain_results[row, col]}({crown_results[row, col]}⭐)" 
                            for col in range(len(terrain_results[row]))]))
        print("\nFuldført! Klassifikationsresultatet er gemt til", args.output)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fejl: {e}")

if __name__ == "__main__":
    main()
