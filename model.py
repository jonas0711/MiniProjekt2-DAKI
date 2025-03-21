import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import json
import pickle

# Konstanter
TILES_DIR = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
LABELS_FILE = "tile_labels_mapping.json"
MODEL_FILE = "kingdomino_terrain_model.pkl"

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

def extract_features_batch(images):
    """
    Udtrækker features fra en liste af billeder.
    
    Args:
        images: Liste af RGB billeder
    
    Returns:
        np.array: Feature-matrix hvor hver række er en feature-vektor for et billede
    """
    features = []
    
    for i, image in enumerate(images):
        if i % 100 == 0:
            print(f"Udtrækker features for billede {i+1}/{len(images)}...")
        
        features.append(extract_features(image))
    
    return np.array(features)

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

def save_model(model, file_path=MODEL_FILE):
    """
    Gemmer modellen til en fil.
    
    Args:
        model: TerrainClassifier-modellen der skal gemmes
        file_path: Sti til outputfilen
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model gemt til: {file_path}")

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

def main():
    """
    Hovedfunktion til at træne terrænklassifikationsmodellen.
    """
    print("Indlæser data...")
    images, labels, terrain_classes = load_data()
    
    print(f"Indlæst {len(images)} billeder med {len(set(labels))} forskellige terrænklasser.")
    print("Terrænklasser:", {k: v for k, v in sorted(terrain_classes.items(), key=lambda x: x[1])})
    
    print("Udtrækker HSV og tekstur features...")
    features = extract_features_batch(images)
    print(f"Udtrukket features med form: {features.shape}")
    
    # Del data i trænings- og testsæt (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("Træner TerrainClassifier model (LDA + KNN)...")
    model = TerrainClassifier(terrain_classes=terrain_classes)
    model.fit(X_train, y_train)
    
    print("Evaluerer model...")
    y_pred = model.predict(X_test)
    
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
    
    # Gem modellen
    print("Gemmer model...")
    save_model(model)
    
    print("Træning og evaluering afsluttet. Model er klar til brug med kingdomino_classifier.py")

if __name__ == "__main__":
    main()