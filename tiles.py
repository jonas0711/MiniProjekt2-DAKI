import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_board_image(image_path):
    """
    Indlæser et billede af et King Domino-bræt.
    
    Args:
        image_path: Sti til billedfilen
    
    Returns:
        Indlæst billede
    """
    # Indlæs billedet
    image = cv2.imread(image_path)
    
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

def visualize_tiles(tiles, grid_size=5):
    """
    Visualiserer alle tiles fra et bræt i et grid.
    
    Args:
        tiles: 2D liste af tiles
        grid_size: Antal tiles i hver retning
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for row in range(grid_size):
        for col in range(grid_size):
            axes[row, col].imshow(tiles[row][col])
            axes[row, col].set_title(f"Tile ({row}, {col})")
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_tiles(tiles, output_dir, board_name):
    """
    Gemmer alle tiles fra et bræt som individuelle billeder.
    
    Args:
        tiles: 2D liste af tiles
        output_dir: Mappe, hvor tiles skal gemmes
        board_name: Navn på brættet (bruges i filnavnet)
    """
    # Opret output-mappen, hvis den ikke eksisterer
    os.makedirs(output_dir, exist_ok=True)
    
    # Gennemgå hver tile og gem den
    for row in range(len(tiles)):
        for col in range(len(tiles[row])):
            # Få den aktuelle tile
            tile = tiles[row][col]
            
            # Konverter fra RGB til BGR (cv2 gemmer som BGR)
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            
            # Definer filnavn
            filename = f"{board_name}_tile_{row}_{col}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Gem billedet
            cv2.imwrite(filepath, tile_bgr)
            
    print(f"Alle tiles gemt i {output_dir}")

def process_board_images(input_dir, output_dir):
    """
    Behandler alle brætbilleder i en mappe og gemmer deres tiles.
    
    Args:
        input_dir: Mappe med brætbilleder
        output_dir: Mappe, hvor tiles skal gemmes
    """
    # Få alle billedfiler i inputmappen
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Opret fuld sti til inputbilledet
        image_path = os.path.join(input_dir, image_file)
        
        # Få board_name ved at fjerne filendelsen
        board_name = os.path.splitext(image_file)[0]
        
        print(f"Behandler {board_name}...")
        
        # Indlæs brætbilledet
        board_image = load_board_image(image_path)
        
        # Opdel brættet i tiles
        tiles = divide_board_into_tiles(board_image)
        
        # Gem alle tiles
        save_tiles(tiles, output_dir, board_name)
        
        print(f"Færdig med {board_name}\n")

# Eksempel på brug
if __name__ == "__main__":
    # Definer input- og output-mapper
    input_dir = "KingDominoDataset/KingDominoDataset/Cropped and perspective corrected boards"
    output_dir = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
    
    # Behandl alle brætbilleder
    process_board_images(input_dir, output_dir)
    
    # Eksempel på at indlæse og visualisere et enkelt bræt (til test)
    # test_image_path = "KingDominoDataset/KingDominoDataset/Cropped and perspective corrected boards/board1.jpg"
    # board_image = load_board_image(test_image_path)
    # tiles = divide_board_into_tiles(board_image)
    # visualize_tiles(tiles)