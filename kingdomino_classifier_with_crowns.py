import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# Importér fra eksisterende kode
from model import load_model, extract_features, TerrainClassifier
from crown_detector import detect_crowns, visualize_crown_detection

# Standard sti til modellen
MODEL_FILE = "kingdomino_terrain_model.pkl"

# Standard sti til plade 1
DEFAULT_IMAGE_PATH = r"KingDominoDataset\KingDominoDataset\Cropped and perspective corrected boards\1.jpg"

def load_board_image(image_path):
    """
    Indlæser et billede af et King Domino-bræt.
    
    Args:
        image_path: Sti til billedfilen
    
    Returns:
        Indlæst billede i RGB-format
    """
    # Kontroller at filen eksisterer
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Billedfilen {image_path} findes ikke.")
    
    # Indlæs billedet
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Kunne ikke indlæse billedet fra {image_path}.")
    
    # Konverter fra BGR til RGB farveformat (cv2 indlæser som BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_rgb

def divide_board_into_tiles(image, grid_size=5):
    """
    Opdeler et King Domino-bræt i individuelle tiles.
    
    Args:
        image: Indlæst billede af brættet
        grid_size: Antal tiles i hver retning (standard: 5x5)
    
    Returns:
        Liste af tiles (2D numpy array med [row][col])
    """
    height, width, _ = image.shape
    
    # Beregn størrelsen af hver tile
    tile_height = height // grid_size
    tile_width = width // grid_size
    
    # Opret en 2D liste til at gemme tiles
    tiles = []
    
    # Gennemgå hver række og kolonne for at udtrække tiles
    for row in range(grid_size):
        tile_row = []
        for col in range(grid_size):
            # Beregn startpositionen for denne tile
            y_start = row * tile_height
            x_start = col * tile_width
            
            # Udskær tile fra det originale billede
            tile = image[y_start:y_start + tile_height, x_start:x_start + tile_width]
            tile_row.append(tile)
        
        tiles.append(tile_row)
    
    return tiles

def classify_tiles_with_crowns(tiles, model):
    """
    Klassificerer et grid af tiles ved hjælp af den indlæste model og detekterer kroner.
    
    Args:
        tiles: 2D liste af tile-billeder
        model: Indlæst TerrainClassifier model
    
    Returns:
        tuple: (terrænresultater, kroneresultater)
    """
    grid_size = len(tiles)
    
    # Opret arrays til at gemme resultater
    terrain_results = np.empty((grid_size, grid_size), dtype=object)
    crown_results = np.empty((grid_size, grid_size), dtype=int)
    
    # Forbered features for alle tiles
    all_features = []
    tile_positions = []
    
    # Udtrækker features for terrænklassifikation
    for row in range(grid_size):
        for col in range(grid_size):
            # Udtræk features fra det nuværende tile
            tile_features = extract_features(tiles[row][col])
            
            # Konverter til NumPy array hvis det ikke allerede er det
            if not isinstance(tile_features, np.ndarray):
                tile_features = np.array(tile_features)
            
            # Gem features og position
            all_features.append(tile_features)
            tile_positions.append((row, col))
    
    # Konverter til NumPy array
    all_features = np.array(all_features)
    
    # Klassificer alle features på én gang med den eksisterende model
    terrain_types = model.predict_terrain(all_features)
    
    # Placer klassifikationsresultater i output-grid og detekter kroner for hvert tile
    for i, (row, col) in enumerate(tile_positions):
        terrain_type = terrain_types[i]
        terrain_results[row, col] = terrain_type
        
        # Spring over kronedetektering for Home og Unknown terræn
        if terrain_type in ['Home', 'Unknown']:
            crown_results[row, col] = 0
            continue
        
        # Detekter kroner baseret på terræntype med vores nye funktion
        crown_count = detect_crowns(tiles[row][col], terrain_type)
        crown_results[row, col] = crown_count
    
    return terrain_results, crown_results

def visualize_classification_with_crowns(original_image, tiles, terrain_results, crown_results, output_path=None):
    """
    Visualiserer klassifikationsresultaterne med kroner.
    
    Args:
        original_image: Det originale pladebillede
        tiles: 2D liste af tile-billeder
        terrain_results: 2D array med terræntype-strenge
        crown_results: 2D array med antal kroner
        output_path: Sti til at gemme visualiseringen (None = vis interaktivt)
    """
    grid_size = len(tiles)
    
    # Definer farver for forskellige terræntyper
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
    
    # 1. Visualiser med farvekodet overlay
    plt.figure(figsize=(15, 10))
    
    # Vis original billede
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original plade")
    plt.axis('off')
    
    # Vis klassifikationsresultat
    plt.subplot(1, 2, 2)
    
    # Lav et blankt billede til at tegne klassifikationen på
    classification_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    # Farvelæg hver celle baseret på den klassificerede terræntype
    for row in range(grid_size):
        for col in range(grid_size):
            terrain = terrain_results[row, col]
            color = terrain_colors.get(terrain, 'white')  # Standard hvid hvis terræntypen ikke findes
            
            # Konverter farvenavn til RGB-tuple for NumPy array
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
                rgb = [255, 255, 255]  # hvid
                
            classification_img[row, col] = rgb
    
    # Vis det farvekodede klassifikationsresultat
    plt.imshow(classification_img)
    
    # Tilføj terrænlabels og kroner til hver celle
    for row in range(grid_size):
        for col in range(grid_size):
            terrain = terrain_results[row, col]
            crowns = crown_results[row, col]
            
            # Vis terræntype og antal kroner
            crown_text = '⭐' * crowns if crowns > 0 else ''
            label_text = f"{terrain}\n{crown_text}"
            
            plt.text(col, row, label_text, 
                     ha="center", va="center", 
                     color="white", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5))
    
    plt.title("Klassifikation med kroner")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 2. Detaljeret grid-visning
    plt.figure(figsize=(15, 15))
    for row in range(grid_size):
        for col in range(grid_size):
            plt.subplot(grid_size, grid_size, row * grid_size + col + 1)
            plt.imshow(tiles[row][col])
            
            # Vis terræntype og antal kroner i titlen
            crown_text = '⭐' * crown_results[row, col] if crown_results[row, col] > 0 else 'Ingen kroner'
            plt.title(f"{terrain_results[row, col]}\n{crown_text}", fontsize=9)
            plt.axis('off')
    
    plt.suptitle("Detaljeret klassifikation med kroner", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Gem eller vis
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisering gemt til {output_path}")
    else:
        plt.show()

def main():
    # Parse command-line argumenter
    parser = argparse.ArgumentParser(description='Klassificer King Domino-plade med kroner')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE_PATH, 
                        help='Sti til billedet af spillepladen')
    parser.add_argument('--model', type=str, default=MODEL_FILE, 
                        help='Sti til den gemte model')
    parser.add_argument('--output', type=str, default='classification_with_crowns.png', 
                        help='Sti til at gemme visualisering')
    parser.add_argument('--grid_size', type=int, default=5, 
                        help='Størrelse af grid (standard: 5)')
    
    args = parser.parse_args()
    
    try:
        # Indlæs model
        print(f"Indlæser model fra {args.model}...")
        model = load_model(args.model)
        
        # Indlæs og forbered pladebilledet
        print(f"Indlæser pladebillede fra {args.image}...")
        board_image = load_board_image(args.image)
        
        # Opdel i tiles
        print(f"Opdeler pladen i {args.grid_size}x{args.grid_size} tiles...")
        tiles = divide_board_into_tiles(board_image, args.grid_size)
        
        # Klassificer hver tile og detekter kroner
        print("Klassificerer tiles og detekterer kroner...")
        terrain_results, crown_results = classify_tiles_with_crowns(tiles, model)
        
        # Vis resultater
        print("Visualiserer resultater...")
        visualize_classification_with_crowns(board_image, tiles, terrain_results, crown_results, args.output)
        
        # Udskriv tekstbaseret resultat til konsollen
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