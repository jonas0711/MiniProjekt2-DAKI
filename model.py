import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import json
from skimage.feature import graycomatrix, graycoprops

# Konstanter
TILES_DIR = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
LABELS_FILE = "tile_labels_mapping.json"

def load_data():
    """
    Indlæser tile-billeder og deres tilhørende labels.
    
    Returns:
        tuple: (billeder, labels, terrain_klasser)
            - billeder: Liste med tile-billeder
            - labels: Liste med terrain-klasselabels
            - terrain_klasser: Dictionary, der mapper terræntyper til numeriske labels
    """
    # Tjek om labels-filen eksisterer
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels-filen {LABELS_FILE} blev ikke fundet. Kør labels.py først.")
    
    # Indlæs labels
    with open(LABELS_FILE, 'r') as f:
        labels_data = json.load(f)
    
    images = []
    terrain_labels = []
    
    # Indsaml unikke terræntyper
    unique_terrains = set()
    
    # Behandl hver bræt og dets tiles
    for board_name, tiles in labels_data.items():
        for tile_pos, tile_info in tiles.items():
            # Hent filsti
            file_path = os.path.join(TILES_DIR, tile_info["filename"])
            
            # Tjek om filen eksisterer
            if not os.path.exists(file_path):
                print(f"Advarsel: Filen {file_path} blev ikke fundet. Springer over.")
                continue
            
            # Indlæs billede
            img = cv2.imread(file_path)
            if img is None:
                print(f"Advarsel: Kunne ikke indlæse billedet {file_path}. Springer over.")
                continue
            
            # Konverter fra BGR til RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Hent terræntype (spring over hvis den er ukendt eller speciel)
            terrain = tile_info["terrain"]
            if terrain in ["Unknown", "Home", "Table"]:
                continue
            
            # Tilføj til data
            images.append(img)
            terrain_labels.append(terrain)
            
            # Tilføj til unikke terræntyper
            unique_terrains.add(terrain)
    
    # Lav terrain-til-klasse mapping
    terrain_classes = {terrain: i for i, terrain in enumerate(sorted(unique_terrains))}
    
    # Konverter terrain-labels til numeriske klasselabels
    labels = [terrain_classes[terrain] for terrain in terrain_labels]
    
    return images, np.array(labels), terrain_classes

def extract_rgb_histogram(image, bins=32):
    """
    Udtrækker RGB histogram features fra et billede.
    
    Args:
        image: RGB billede
        bins: Antal bins for hver farvekanal
    
    Returns:
        np.array: Sammenkædet normaliseret RGB histogram (3*bins features)
    """
    # Konverter fra BGR til RGB hvis nødvendigt (safety check)
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_rgb = image
    else:
        print("Advarsel: Billede er ikke i RGB format")
        return np.zeros(bins * 3)  # Returner tom vektor i tilfælde af fejl
    
    # Beregn histogrammer for hver kanal
    hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256])
    
    # Normaliser histogrammer
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    
    # Sammenkæd histogrammer
    return np.concatenate([hist_r, hist_g, hist_b])

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

def extract_features(images):
    """
    Udtrækker features fra billeder.
    
    Args:
        images: Liste af RGB billeder
    
    Returns:
        np.array: Feature-matrix hvor hver række er en feature-vektor for et billede
    """
    features = []
    
    for image in images:
        # Udtrækker RGB histogram features
        rgb_hist = extract_rgb_histogram(image)
        
        # Udtrækker HSV histogram features
        hsv_hist = extract_hsv_histogram(image)
        
        # Udtrækker tekstur features
        texture_hist = extract_texture_histogram(image)
        
        # Kombinerer features
        combined_features = np.concatenate([rgb_hist, hsv_hist, texture_hist])
        
        features.append(combined_features)
    
    return np.array(features)

def apply_lda(features, labels, n_components=None):
    """
    Anvender Linear Discriminant Analysis til dimensionsreduktion.
    
    Args:
        features: Feature-matrix
        labels: Klasselabels
        n_components: Antal LDA komponenter at beholde (default: min(n_classes-1, n_features))
    
    Returns:
        tuple: (lda, transformed_features)
            - lda: Fitted LDA model
            - transformed_features: LDA-transformerede features
    """
    # Opret LDA model
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    
    # Fit og transformer features
    transformed_features = lda.fit_transform(features, labels)
    
    return lda, transformed_features

def visualize_lda(lda_features, labels, terrain_classes):
    """
    Visualiserer data i LDA space.
    
    Args:
        lda_features: LDA-transformerede features
        labels: Klasselabels
        terrain_classes: Dict mapping af terræntyper til numeriske labels
    """
    # Opret omvendt mapping fra numeriske labels til terræntyper
    terrain_names = {v: k for k, v in terrain_classes.items()}
    
    # Opret plot
    plt.figure(figsize=(12, 10))
    
    # Hvis vi har mindst 2 LDA komponenter
    if lda_features.shape[1] >= 2:
        # Scatter plot af de første to LDA komponenter
        for label in np.unique(labels):
            plt.scatter(
                lda_features[labels == label, 0],
                lda_features[labels == label, 1],
                label=terrain_names[label]
            )
        
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.title('LDA Transformation')
        plt.legend()
        plt.savefig('lda_visualization.png')
        print("LDA visualisering gemt til lda_visualization.png")
    else:
        print("Ikke nok LDA komponenter til visualisering.")

def train_knn(X_train, y_train, k=5):
    """
    Træner en K-Nearest Neighbors klassifikator.
    
    Args:
        X_train: Trænings-features
        y_train: Trænings-labels
        k: Antal naboer
    
    Returns:
        KNeighborsClassifier: Trænet KNN model
    """
    # Opret KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Træn model
    knn.fit(X_train, y_train)
    
    return knn

def main():
    """
    Hovedfunktion til at køre hele pipeline'en.
    """
    print("Indlæser data...")
    images, labels, terrain_classes = load_data()
    
    print(f"Indlæst {len(images)} billeder med {len(set(labels))} forskellige terrænklasser.")
    print("Terrænklasser:", {k: v for k, v in sorted(terrain_classes.items(), key=lambda x: x[1])})
    
    print("Udtrækker features...")
    features = extract_features(images)
    print(f"Udtrukket features med form: {features.shape}")
    
    # Del data i trænings- og testsæt (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("Anvender LDA...")
    # Antal komponenter bør højst være antal klasser - 1
    n_components = min(len(set(labels)) - 1, X_train.shape[1])
    lda, X_train_lda = apply_lda(X_train, y_train, n_components=n_components)
    
    # Transformer test features
    X_test_lda = lda.transform(X_test)
    
    print(f"LDA reducerede features til form: {X_train_lda.shape}")
    
    # Visualiser LDA space
    visualize_lda(X_train_lda, y_train, terrain_classes)
    
    print("Træner KNN klassifikator...")
    knn = train_knn(X_train_lda, y_train, k=5)
    
    # Evaluer model
    y_pred = knn.predict(X_test_lda)
    
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
    plt.savefig('confusion_matrix.png')
    
    print("Resultater gemt til confusion_matrix.png")

if __name__ == "__main__":
    main()