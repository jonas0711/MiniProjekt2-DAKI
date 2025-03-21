import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import pickle
from collections import deque

# Definer standardstier
TILES_DIR = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
LABELS_FILE = "Excel+JSON/tile_labels_mapping.json"
MODEL_FILE = "kingdomino_terrain_model.pkl"
DEFAULT_IMAGE_PATH = "KingDominoDataset/KingDominoDataset/Cropped and perspective corrected boards/1.jpg"
DEFAULT_OUTPUT_PATH = "kingdomino_result.png"

# VIGTIGT: Vi skal definere TerrainClassifier-klassen for at Pickle kan deserialisere modellen
class TerrainClassifier:
    """
    Kingdomino terrænklassifikator, der kombinerer LDA og KNN i en enkelt model.
    """
    
    def __init__(self, lda=None, knn=None, terrain_classes=None):
        """
        Initialiserer klassifikatoren.
        
        Args:
            lda: LDA model (optional)
            knn: KNN model (optional)
            terrain_classes: Dictionary med mapping af terræntyper til numeriske labels
        """
        self.lda = lda
        self.knn = knn
        self.terrain_classes = terrain_classes
        self.is_fitted = (lda is not None and knn is not None)
    
    def fit(self, X, y):
        """
        Træner modellen på features og labels.
        
        Args:
            X: Feature-matrix
            y: Klasse-labels
        
        Returns:
            self: Trænet model
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.neighbors import KNeighborsClassifier
        
        # Bestem antal komponenter (maks antal klasser - 1)
        n_components = min(len(np.unique(y)), X.shape[1]) - 1
        
        # Træn LDA
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = self.lda.fit_transform(X, y)
        
        # Træn KNN på LDA-features
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(X_lda, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Forudsiger klasser for features.
        
        Args:
            X: Feature-matrix eller enkelt feature-vektor
        
        Returns:
            np.array: Forudsagte klasser
        """
        if not self.is_fitted:
            raise ValueError("Modellen er ikke trænet endnu.")
        
        # Kontroller om X er en enkelt feature-vektor eller et batch
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Transform med LDA
        X_lda = self.lda.transform(X)
        
        # Forudsig med KNN
        return self.knn.predict(X_lda)
    
    def predict_terrain(self, X):
        """
        Forudsiger terræntyper for features.
        
        Args:
            X: Feature-matrix eller enkelt feature-vektor
        
        Returns:
            list: Forudsagte terræntyper
        """
        if self.terrain_classes is None:
            raise ValueError("Terrain classes mapping er ikke tilgængelig.")
        
        # Forudsig numeriske labels
        y_pred = self.predict(X)
        
        # Konverter til terræntyper
        terrain_names = {v: k for k, v in self.terrain_classes.items()}
        return [terrain_names[label] for label in y_pred]

def extract_hsv_histogram(image, bins=32):
    """
    Udtrækker HSV histogram features fra et billede.
    
    Args:
        image: RGB billede
        bins: Antal bins for hver HSV kanal
    
    Returns:
        np.array: Sammenkædet normaliseret HSV histogram (3*bins features)
    """
    # Konverter til HSV
    try:
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    except:
        print("Advarsel: Kunne ikke konvertere til HSV")
        return np.zeros(bins * 3)  # Returner tom vektor i tilfælde af fejl
    
    # Beregn histogrammer for hver kanal
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    
    # Normaliser histogrammer
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    # Sammenkæd histogrammer
    return np.concatenate([hist_h, hist_s, hist_v])

def extract_texture_histogram(image, bins=9):
    """
    Udtrækker tekstur features ved hjælp af gradient orienteringer.
    
    Args:
        image: RGB billede
        bins: Antal orienterings-bins
    
    Returns:
        np.array: Histogram over gradient orienteringer
    """
    # Konverter til gråtone
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except:
        print("Advarsel: Kunne ikke konvertere til gråtone")
        return np.zeros(bins)  # Returner tom vektor i tilfælde af fejl
    
    # Beregn gradienter ved hjælp af Sobel-operatoren
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Beregn gradient magnitude og retning
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi) % 180
    
    # Opret histogram
    hist = np.zeros(bins)
    bin_width = 180 / bins
    
    # Akkumuler histogram baseret på gradient retning og magnitude
    for i in range(bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        
        # Opret maske for denne bin
        mask = ((direction >= bin_start) & (direction < bin_end))
        
        # Sum magnitude for denne bin
        hist[i] = np.sum(magnitude[mask])
    
    # Normaliser histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def extract_features(image):
    """
    Udtrækker HSV og tekstur features fra et enkelt billede.
    
    Args:
        image: RGB billede
    
    Returns:
        np.array: Feature-vektor
    """
    # Udtræk HSV histogram features
    hsv_hist = extract_hsv_histogram(image)
    
    # Udtræk tekstur features
    texture_hist = extract_texture_histogram(image)
    
    # Kombiner features
    combined_features = np.concatenate([hsv_hist, texture_hist])
    
    return combined_features

def load_model(file_path=MODEL_FILE):
    """
    Indlæser en gemt model fra en fil.
    
    Args:
        file_path: Sti til modelfilen
    
    Returns:
        TerrainClassifier: Indlæst model
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Modelfilen {file_path} findes ikke.")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model indlæst fra: {file_path}")
    return model

def detect_crowns(tile, terrain_type):
    """
    Detekterer kroner i et tile baseret på terræntypen.
    
    Args:
        tile: RGB-billede af tile
        terrain_type: Type af terræn ('Field', 'Forest', etc.)
        
    Returns:
        int: Antal kroner detekteret
    """
    # Konverter til HSV (bedre til farvebaseret segmentering)
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    
    # Definér HSV-intervaller for kroner baseret på terræntype
    # Disse værdier skal justeres baseret på terrænets farve
    if terrain_type == 'Field':
        # For gult terræn (sværere at skelne) - strengere krav
        lower_gold = np.array([20, 150, 150])  # Mørkere gul/guld
        upper_gold = np.array([35, 255, 255])  # Lysere gul/guld
    else:
        # For andre terræntyper (lettere at skelne)
        lower_gold = np.array([15, 100, 100])  # Bredere interval
        upper_gold = np.array([40, 255, 255])
    
    # Opret maske baseret på farveinterval
    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    
    # Anvend morfologiske operationer for at fjerne støj
    kernel = np.ones((3, 3), np.uint8)
    # Opening (erosion efterfulgt af dilation) fjerner små støjområder
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Closing (dilation efterfulgt af erosion) lukker små huller i objekter
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find konturer i masken
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer konturer baseret på areal og form
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip små konturer (støj)
        if area < 50:
            continue
            
        # Beregn cirkularity (4π × Area / Perimeter²)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter baseret på cirkularity og areal
        # Kroner er typisk nogenlunde cirkulære og har en bestemt størrelse
        if circularity > 0.3 and area > 50 and area < 1000:
            filtered_contours.append(contour)
    
    # For Field (gult terræn), brug ekstra verifikation
    if terrain_type == 'Field' and len(filtered_contours) > 0:
        # For gult terræn: anvend kantdetektion som ekstra verifikation
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        verified_contours = []
        for contour in filtered_contours:
            # Opret en maske for konturen
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Tæl antal kantpixels inden for konturen
            edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
            
            # Hvis der er nok kanter inden for konturen, er det sandsynligvis en krone
            if edge_pixels > 10:
                verified_contours.append(contour)
        
        crown_count = len(verified_contours)
    else:
        crown_count = len(filtered_contours)
    
    # Sikrer at vi ikke overstiger 3 (det maksimale antal kroner på et felt i Kingdomino)
    return min(crown_count, 3)

def load_board_image(image_path):
    """Indlæser et billede af et King Domino-bræt."""
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
    """Opdeler et King Domino-bræt i individuelle tiles."""
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
    """Klassificerer et grid af tiles og detekterer kroner."""
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

def identify_connected_territories(terrain_results, crown_results):
    """
    Identificerer sammenhængende territorier på brættet.
    
    Args:
        terrain_results: 2D array med terræntyper
        crown_results: 2D array med antal kroner
        
    Returns:
        list: Liste af territorier med information om terræntype, felter og kroner
    """
    grid_size = len(terrain_results)
    
    # Opret visited array
    visited = np.zeros((grid_size, grid_size), dtype=bool)
    
    # Liste til at gemme alle territorier
    territories = []
    
    # Retninger for naboceller (4-connectivity: op, ned, venstre, højre)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Gennemløb alle celler
    for row in range(grid_size):
        for col in range(grid_size):
            # Spring over celler, der allerede er besøgt eller er Home/Unknown
            if visited[row, col] or terrain_results[row, col] in ['Home', 'Unknown']:
                continue
            
            # Nuværende terræntype
            current_terrain = terrain_results[row, col]
            
            # BFS til at finde sammenhængende territorium
            territory_tiles = []
            total_crowns = 0
            queue = deque([(row, col)])
            visited[row, col] = True
            
            while queue:
                r, c = queue.popleft()
                territory_tiles.append((r, c))
                total_crowns += crown_results[r, c]
                
                # Tjek alle fire retninger
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    
                    # Tjek om den nye position er gyldig
                    if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                        not visited[new_r, new_c] and 
                        terrain_results[new_r, new_c] == current_terrain):
                        queue.append((new_r, new_c))
                        visited[new_r, new_c] = True
            
            # Tilføj territorium til listen
            territories.append({
                'terrain': current_terrain,
                'tiles': territory_tiles,
                'crowns': total_crowns,
                'score': len(territory_tiles) * total_crowns if total_crowns > 0 else 0
            })
    
    return territories

def score_board(terrain_results, crown_results):
    """
    Beregner den samlede score for brættet baseret på Kingdomino-regler.
    
    Args:
        terrain_results: 2D array med terræntyper
        crown_results: 2D array med antal kroner
        
    Returns:
        dict: Score-resultat med detaljer
    """
    # Find territorier
    territories = identify_connected_territories(terrain_results, crown_results)
    
    # Beregn harmony bonus (fuldstændigt 5x5 grid uden huller)
    grid_size = len(terrain_results)
    has_harmony = True
    for row in range(grid_size):
        for col in range(grid_size):
            if terrain_results[row, col] == 'Unknown':
                has_harmony = False
                break
    
    harmony_bonus = 5 if has_harmony else 0
    
    # Beregn score for hvert territorium og total score
    total_score = sum(territory['score'] for territory in territories) + harmony_bonus
    
    # Opret detaljeret resultat
    result = {
        'territories': territories,
        'harmony_bonus': harmony_bonus,
        'total_score': total_score
    }
    
    return result

def visualize_scored_board(original_image, tiles, terrain_results, crown_results, score_result, output_path=None):
    """Visualiserer det scorede bræt."""
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
    
    # Opret en maske til at markere hvert territorium
    territory_mask = np.zeros((grid_size, grid_size), dtype=int)
    
    for i, territory in enumerate(score_result['territories']):
        for row, col in territory['tiles']:
            territory_mask[row, col] = i + 1
    
    # Opret visualisering
    plt.figure(figsize=(15, 15))
    
    # Vis original billede
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title("Original plade")
    plt.axis('off')
    
    # Vis terrænklassifikation med kroner
    plt.subplot(2, 2, 2)
    terrain_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    for row in range(grid_size):
        for col in range(grid_size):
            terrain = terrain_results[row, col]
            color = terrain_colors.get(terrain, 'white')
            
            # Konverter farvenavn til RGB
            rgb = mcolors.to_rgb(color)
            terrain_img[row, col] = [int(c*255) for c in rgb]
    
    plt.imshow(terrain_img)
    
    # Tilføj terrænlabels og kroner
    for row in range(grid_size):
        for col in range(grid_size):
            terrain = terrain_results[row, col]
            crowns = crown_results[row, col]
            
            crown_text = '⭐' * crowns if crowns > 0 else ''
            plt.text(col, row, f"{terrain}\n{crown_text}", 
                     ha="center", va="center", 
                     color="white", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    
    plt.title("Terrænklassifikation med kroner")
    plt.axis('off')
    
    # Vis territorier med scores
    plt.subplot(2, 2, 3)
    
    # Opret et farvekodet billede baseret på territorier
    cmap = plt.cm.get_cmap('tab20', len(score_result['territories']) + 1)
    territory_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    for row in range(grid_size):
        for col in range(grid_size):
            territory_id = territory_mask[row, col]
            if territory_id > 0:
                rgb = cmap(territory_id)[:3]
                territory_img[row, col] = [int(c*255) for c in rgb]
            else:
                # Home eller Unknown
                terrain = terrain_results[row, col]
                color = terrain_colors.get(terrain, 'white')
                rgb = mcolors.to_rgb(color)
                territory_img[row, col] = [int(c*255) for c in rgb]
    
    plt.imshow(territory_img)
    
    # Tilføj territorium scores til hvert territorium
    territory_scores = {}
    for i, territory in enumerate(score_result['territories']):
        # Beregn centroid for territoriet
        centroid_row = sum(row for row, _ in territory['tiles']) / len(territory['tiles'])
        centroid_col = sum(col for _, col in territory['tiles']) / len(territory['tiles'])
        
        territory_scores[(centroid_row, centroid_col)] = territory['score']
    
    # Tilføj scores til visualiseringen
    for (row, col), score in territory_scores.items():
        plt.text(col, row, f"{score}", 
                 ha="center", va="center", 
                 color="white", fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="circle,pad=0.3", fc="black", alpha=0.7))
    
    plt.title("Territorier med scores")
    plt.axis('off')
    
    # Vis samlet score information
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Tekst med score detaljer
    score_text = f"KINGDOMINO SCORE\n\n"
    
    # Sorter territorier efter score
    sorted_territories = sorted(score_result['territories'], key=lambda t: t['score'], reverse=True)
    
    for i, territory in enumerate(sorted_territories):
        terrain = territory['terrain']
        tiles = len(territory['tiles'])
        crowns = territory['crowns']
        score = territory['score']
        
        score_text += f"Territorium {i+1} ({terrain}):\n"
        score_text += f"  {tiles} felter × {crowns} kroner = {score} points\n\n"
    
    score_text += f"Harmony Bonus: {score_result['harmony_bonus']} points\n"
    score_text += f"Total Score: {score_result['total_score']} points"
    
    plt.text(0.5, 0.5, score_text, 
             ha="center", va="center", 
             fontsize=12,
             bbox=dict(boxstyle="round,pad=1", fc="white", alpha=0.9))
    
    plt.tight_layout()
    
    # Gem eller vis
    if output_path:
        plt.savefig(output_path)
        print(f"Score visualisering gemt til {output_path}")
    else:
        plt.show()

def full_pipeline(image_path=DEFAULT_IMAGE_PATH, model_path=MODEL_FILE, output_path=DEFAULT_OUTPUT_PATH, grid_size=5):
    """
    Kører den komplette pipeline: 
    1. Klassificerer terræn
    2. Detekterer kroner
    3. Beregner score
    4. Visualiserer resultat
    """
    try:
        # Indlæs model
        print(f"Indlæser model fra {model_path}...")
        model = load_model(model_path)
        
        # Indlæs pladebilledet
        print(f"Indlæser pladebillede fra {image_path}...")
        board_image = load_board_image(image_path)
        
        # Opdel i tiles
        print(f"Opdeler pladen i {grid_size}x{grid_size} tiles...")
        tiles = divide_board_into_tiles(board_image, grid_size)
        
        # Klassificer hver tile og detekter kroner
        print("Klassificerer tiles og detekterer kroner...")
        terrain_results, crown_results = classify_tiles_with_crowns(tiles, model)
        
        # Beregn score
        print("Beregner score...")
        score_result = score_board(terrain_results, crown_results)
        
        # Vis resultater
        print("Visualiserer resultater...")
        visualize_scored_board(board_image, tiles, terrain_results, crown_results, score_result, output_path)
        
        print("\nKlassifikationsresultat med kroner:")
        for row in range(len(terrain_results)):
            print(" ".join([f"{terrain_results[row, col]}({crown_results[row, col]}⭐)" 
                            for col in range(len(terrain_results[row]))]))
        
        print(f"\nTotal score: {score_result['total_score']} points")
        print(f"\nFuldført! Resultat gemt til {output_path}")
        
        return score_result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fejl: {e}")
        return None

def main():
    # Parse command-line argumenter (men alle har standardværdier nu)
    parser = argparse.ArgumentParser(description='Kør Kingdomino analyse pipeline')
    parser.add_argument('--image', default=DEFAULT_IMAGE_PATH, help='Sti til billedet af spillepladen')
    parser.add_argument('--model', default=MODEL_FILE, help='Sti til den gemte model')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_PATH, help='Sti til at gemme visualisering')
    parser.add_argument('--grid_size', type=int, default=5, help='Størrelse af grid (standard: 5)')
    
    args = parser.parse_args()
    
    # Kør fuld pipeline
    full_pipeline(args.image, args.model, args.output, args.grid_size)

if __name__ == "__main__":
    main()