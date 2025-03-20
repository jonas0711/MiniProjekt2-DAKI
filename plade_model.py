import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import til 3D-plotting
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re
import json

# Konstanter
TILES_DIR = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
LABELS_FILE = "tile_labels_mapping.json"

def extract_board_number(filename):
    """
    Udtrækker pladenummeret fra et filnavn.
    """
    match = re.search(r'^(\d+)_tile_', filename)
    if match:
        return int(match.group(1))
    return None

def load_data_by_board_split(tiles_dir, labels_file, train_boards=range(1, 60), test_boards=range(60, 75)):
    """
    Indlæser data opdelt efter pladenumre.
    """
    # Tjek om labels-filen eksisterer
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels-filen {labels_file} blev ikke fundet.")
    
    # Indlæs labels fra JSON
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    
    # Opret lister til at gemme trænings- og testdata
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Mængde til at gemme alle unikke terræntyper
    unique_terrains = set()
    
    # Behandl hver plade og dens tiles
    for board_name, tiles in labels_data.items():
        # Forsøg at udtrække pladenummer fra pladenavn
        try:
            board_number = int(board_name)
        except ValueError:
            print(f"Advarsel: Pladenavn '{board_name}' er ikke et gyldigt pladenummer. Springer over.")
            continue
        
        # Tjek om denne plade er i trænings- eller testsættet
        is_train = board_number in train_boards
        is_test = board_number in test_boards
        
        if not (is_train or is_test):
            continue
        
        # Behandl hver tile
        for tile_pos, tile_info in tiles.items():
            # Hent tile-filnavn
            tile_file = tile_info["filename"]
            
            # Hent terræntype
            terrain = tile_info["terrain"]
            
            # Spring specielle terræntyper over
            if terrain in ["Unknown", "Home", "Table"]:
                continue
            
            # Indlæs tile-billedet
            tile_path = os.path.join(tiles_dir, tile_file)
            if not os.path.exists(tile_path):
                print(f"Advarsel: Tile-billede {tile_path} blev ikke fundet. Springer over.")
                continue
            
            tile_image = cv2.imread(tile_path)
            if tile_image is None:
                print(f"Advarsel: Kunne ikke indlæse tile-billede {tile_path}. Springer over.")
                continue
            
            # Konverter fra BGR til RGB
            tile_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
            
            # Tilføj terræntype til unikke terræntyper
            unique_terrains.add(terrain)
            
            # Tilføj til passende datasæt
            if is_train:
                train_images.append(tile_image)
                train_labels.append(terrain)
            elif is_test:
                test_images.append(tile_image)
                test_labels.append(terrain)
    
    # Opret terræntype-til-klasse mapping
    terrain_classes = {terrain: i for i, terrain in enumerate(sorted(unique_terrains))}
    
    # Konverter labels til numeriske klasseindeks
    train_labels = [terrain_classes[label] for label in train_labels]
    test_labels = [terrain_classes[label] for label in test_labels]
    
    print(f"Indlæst {len(train_images)} træningsbilleder og {len(test_images)} testbilleder.")
    print(f"Fundet {len(terrain_classes)} unikke terrænklasser: {terrain_classes}")
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels), terrain_classes

def extract_rgb_histogram(image, bins=32):
    """
    Udtrækker RGB histogram features fra et billede.
    """
    # Beregn histogrammer for hver kanal
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    # Normaliser histogrammer
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    
    # Sammenkæd histogrammer
    return np.concatenate([hist_r, hist_g, hist_b])

def extract_hsv_histogram(image, bins=32):
    """
    Udtrækker HSV histogram features fra et billede.
    """
    # Konverter til HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
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
    """
    # Konverter til gråtone
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
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

def extract_features(images):
    """
    Udtrækker features fra billeder.
    """
    features = []
    
    for i, image in enumerate(images):
        if i % 100 == 0:
            print(f"Udtrækker features for billede {i+1}/{len(images)}...")
        
        # Udtræk RGB histogram features
        rgb_hist = extract_rgb_histogram(image)
        
        # Udtræk HSV histogram features
        hsv_hist = extract_hsv_histogram(image)
        
        # Udtræk tekstur features
        texture_hist = extract_texture_histogram(image)
        
        # Kombiner features
        combined_features = np.concatenate([rgb_hist, hsv_hist, texture_hist])
        
        features.append(combined_features)
    
    return np.array(features)

def apply_lda(X_train, y_train, X_test, n_components=None):
    """
    Anvender Linear Discriminant Analysis til trænings- og testdata.
    """
    # Sikrer vi bruger minimum 3 komponenter hvis muligt
    if n_components is None:
        n_components = min(len(np.unique(y_train)) - 1, X_train.shape[1])
    else:
        n_components = min(n_components, len(np.unique(y_train)) - 1, X_train.shape[1])
    
    # Opret LDA model
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    
    # Fit på træningsdata
    X_train_lda = lda.fit_transform(X_train, y_train)
    
    # Transformer testdata
    X_test_lda = lda.transform(X_test)
    
    # Print explained variance ratio hvis den er tilgængelig
    if hasattr(lda, 'explained_variance_ratio_'):
        print("Explained variance ratio for LDA komponenter:")
        for i, var in enumerate(lda.explained_variance_ratio_):
            print(f"LD{i+1}: {var:.4f}")
    
    return lda, X_train_lda, X_test_lda

def visualize_lda_3d(X_lda, y, terrain_classes, title="3D LDA Projektion", filename="lda_3d_visualization.png"):
    """
    Visualiserer data i 3D LDA space.
    """
    # Opret omvendt mapping fra klasseindeks til terrænnavne
    terrain_names = {v: k for k, v in terrain_classes.items()}
    
    # Tjek om vi har nok komponenter til 3D-visualisering
    if X_lda.shape[1] < 3:
        print(f"Advarsel: Kun {X_lda.shape[1]} LDA komponenter tilgængelige. Minimum 3 krævet for 3D-visualisering.")
        return
    
    # Opret 3D-plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Definer farver for hver klasse
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    # Plot hver klasse
    for label in np.unique(y):
        if label < len(colors):  # Sikrer at vi ikke løber tør for farver
            mask = y == label
            ax.scatter(
                X_lda[mask, 0],
                X_lda[mask, 1],
                X_lda[mask, 2],
                c=colors[label],
                label=terrain_names[label],
                alpha=0.7
            )
    
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    ax.set_title(title)
    ax.legend()
    
    # Gem figur
    plt.savefig(filename)
    print(f"3D LDA visualisering gemt til {filename}")
    
    # Opret også 2D-visualiseringer fra forskellige vinkler
    # LD1 vs. LD2
    plt.figure(figsize=(10, 8))
    for label in np.unique(y):
        if label < len(colors):
            mask = y == label
            plt.scatter(
                X_lda[mask, 0],
                X_lda[mask, 1],
                c=colors[label],
                label=terrain_names[label],
                alpha=0.7
            )
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(f"{title} - LD1 vs. LD2")
    plt.legend()
    plt.savefig(filename.replace('.png', '_ld1_ld2.png'))
    
    # LD1 vs. LD3
    plt.figure(figsize=(10, 8))
    for label in np.unique(y):
        if label < len(colors):
            mask = y == label
            plt.scatter(
                X_lda[mask, 0],
                X_lda[mask, 2],
                c=colors[label],
                label=terrain_names[label],
                alpha=0.7
            )
    plt.xlabel('LD1')
    plt.ylabel('LD3')
    plt.title(f"{title} - LD1 vs. LD3")
    plt.legend()
    plt.savefig(filename.replace('.png', '_ld1_ld3.png'))
    
    # LD2 vs. LD3
    plt.figure(figsize=(10, 8))
    for label in np.unique(y):
        if label < len(colors):
            mask = y == label
            plt.scatter(
                X_lda[mask, 1],
                X_lda[mask, 2],
                c=colors[label],
                label=terrain_names[label],
                alpha=0.7
            )
    plt.xlabel('LD2')
    plt.ylabel('LD3')
    plt.title(f"{title} - LD2 vs. LD3")
    plt.legend()
    plt.savefig(filename.replace('.png', '_ld2_ld3.png'))

def train_knn(X_train, y_train, k=5):
    """
    Træner en K-Nearest Neighbors klassifikator.
    """
    # Opret KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Træn model
    knn.fit(X_train, y_train)
    
    return knn

def evaluate_model(model, X_test, y_test, terrain_classes):
    """
    Evaluerer en trænet model på testdata.
    """
    # Lav forudsigelser
    y_pred = model.predict(X_test)
    
    # Beregn metrikker
    print("Klassifikationsrapport:")
    print(classification_report(y_test, y_pred))
    
    print("Konfusionsmatrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Map numeriske labels tilbage til terræntyper
    terrain_names = {v: k for k, v in terrain_classes.items()}
    terrain_labels = [terrain_names[i] for i in range(len(terrain_names))]
    
    # Plot konfusionsmatrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Konfusionsmatrix')
    plt.colorbar()
    tick_marks = np.arange(len(terrain_labels))
    plt.xticks(tick_marks, terrain_labels, rotation=90)
    plt.yticks(tick_marks, terrain_labels)
    plt.tight_layout()
    plt.ylabel('Faktisk label')
    plt.xlabel('Forudsagt label')
    plt.savefig('confusion_matrix_board_split.png')
    
    print("Resultater gemt til confusion_matrix_board_split.png")
    
    # Returner nøjagtighed til reference
    return np.mean(y_pred == y_test)

def main():
    """
    Hovedfunktion til at køre hele pipeline'en.
    """
    # Indlæs data opdelt efter plade
    print("Indlæser data...")
    X_train, y_train, X_test, y_test, terrain_classes = load_data_by_board_split(
        TILES_DIR, 
        LABELS_FILE,
        train_boards=range(1, 60),
        test_boards=range(60, 75)
    )
    
    print(f"Træningsdata form: {X_train.shape}")
    print(f"Testdata form: {X_test.shape}")
    
    # Udtræk features
    print("Udtrækker features...")
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)
    
    print(f"Trænings-features form: {X_train_features.shape}")
    print(f"Test-features form: {X_test_features.shape}")
    
    # Anvend LDA med min. 3 komponenter (hvis muligt)
    print("Anvender LDA...")
    n_components = min(len(set(y_train)) - 1, X_train_features.shape[1], 5)  # Max 5 komponenter
    lda, X_train_lda, X_test_lda = apply_lda(X_train_features, y_train, X_test_features, n_components)
    
    print(f"LDA-transformeret træningsdata form: {X_train_lda.shape}")
    print(f"LDA-transformeret testdata form: {X_test_lda.shape}")
    
    # Visualiser LDA projektion i 3D
    visualize_lda_3d(X_train_lda, y_train, terrain_classes, 
                    title="3D LDA Projektion af Træningsdata", 
                    filename="lda_3d_board_split_train.png")
    
    # Træn KNN klassifikator
    print("Træner KNN klassifikator...")
    knn = train_knn(X_train_lda, y_train, k=5)
    
    # Evaluer model
    print("Evaluerer model...")
    accuracy = evaluate_model(knn, X_test_lda, y_test, terrain_classes)
    
    print(f"Test nøjagtighed: {accuracy:.4f}")

if __name__ == "__main__":
    main()