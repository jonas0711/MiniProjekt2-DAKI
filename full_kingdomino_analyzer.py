import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import pickle
from pathlib import Path
import random
from sklearn.metrics import classification_report, confusion_matrix

# Importér klasser og funktioner fra model.py
# Dette er vigtigt for at sikre at TerrainClassifier er i scope når vi deserialiserer
from model import TerrainClassifier, load_model, extract_features, extract_hsv_histogram, extract_texture_histogram

# Constants
TERRAIN_MODEL_FILE = "kingdomino_terrain_model.pkl"
CROWN_TEMPLATES_DIR = "KingDominoDataset/Crown_Templates"
TERRAIN_CATEGORIES_DIR = "KingDominoDataset/TerrainCategories"
TERRAIN_TYPES = ["Field", "Forest", "Grassland", "Lake", "Mine", "Swamp"]
CROWN_TEMPLATE_NAMES = ["Up", "Down", "Left", "Right"]

def load_terrain_model(model_path=TERRAIN_MODEL_FILE):
    """
    Indlæser den gemte terrænklassifikationsmodel.
    
    Args:
        model_path: Sti til modelfillen
        
    Returns:
        Den indlæste model
    """
    try:
        # Sikrer at vi er i samme directory som model.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_full_path = os.path.join(current_dir, model_path)
        
        print(f"Forsøger at indlæse model fra: {model_full_path}")
        with open(model_full_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Terrænklassifikationsmodel indlæst fra {model_full_path}")
        return model
    except Exception as e:
        print(f"Fejl ved indlæsning af model: {e}")
        print("Detaljeret fejl:", e.__class__.__name__)
        import traceback
        traceback.print_exc()
        return None

def load_crown_templates(templates_dir=CROWN_TEMPLATES_DIR):
    """
    Indlæser crown templates fra den angivne mappe.
    
    Args:
        templates_dir: Sti til mappen med crown templates
        
    Returns:
        Tuple med (templates, template_navne)
    """
    template_paths = [
        os.path.join(templates_dir, "Crown_up.png"),
        os.path.join(templates_dir, "Crown_down.png"),
        os.path.join(templates_dir, "Crown_left.png"),
        os.path.join(templates_dir, "Crown_right.png")
    ]
    
    templates = []
    for path in template_paths:
        try:
            # Indlæs template og konverter til gråskala
            template = cv2.imread(path)
            if template is None:
                raise FileNotFoundError(f"Kunne ikke indlæse {path}")
                
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            templates.append(template_gray)
            print(f"Indlæst template: {path}, Form: {template_gray.shape}")
        except Exception as e:
            print(f"Advarsel: {e}")
    
    return templates, CROWN_TEMPLATE_NAMES

def extract_info_from_filename(filename):
    """
    Udtrækker terræntype og antal kroner fra filnavnet.
    
    Args:
        filename: Fuldt filnavn at analysere
        
    Returns:
        Tuple med (terræntype, antal_kroner)
    """
    # Udtræk terræntype fra filnavnet
    terrain_match = re.search(r'_([A-Za-z]+)_(\d+)crowns\.png$', filename)
    if terrain_match:
        terrain_type = terrain_match.group(1)
        crowns = int(terrain_match.group(2))
        return terrain_type, crowns
    
    return None, 0

def detect_crowns_in_tile(tile_image, templates, template_names, threshold=0.6):
    """
    Detekterer kroner i et tile ved hjælp af template matching.
    
    Args:
        tile_image: Billedet af feltet der skal undersøges
        templates: Liste af templates (kroner i forskellige orienteringer)
        template_names: Liste af tilsvarende template navne
        threshold: Minimum værdi for at acceptere et match (0.0-1.0)
    
    Returns:
        Antal detekterede kroner og liste med match-detaljer
    """
    # Konverter til RGB hvis billedet er i BGR-format
    if tile_image.shape[2] == 3 and len(tile_image.shape) == 3:
        # Tjek om billedet allerede er i RGB-format
        try:
            tile_rgb = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
        except:
            tile_rgb = tile_image  # Antag at det allerede er i RGB
    else:
        tile_rgb = tile_image
    
    # Konverter til gråskala for template matching
    tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    
    best_matches = []
    
    # For hver template (op, ned, venstre, højre)
    for template, template_name in zip(templates, template_names):
        # Udfør template matching med Normalized Cross-Correlation
        result = cv2.matchTemplate(tile_gray, template, cv2.TM_CCOEFF_NORMED)
        
        # Find lokationer over threshold
        locations = np.where(result >= threshold)
        
        # Gem matches
        for pt in zip(*locations[::-1]):  # Konverter (y, x) til (x, y)
            match_value = result[pt[1], pt[0]]
            best_matches.append({
                'value': match_value,
                'position': pt,
                'template': template_name,
                'template_size': template.shape
            })
    
    # Sortér matches efter match-værdi (højeste først)
    best_matches.sort(key=lambda x: x['value'], reverse=True)
    
    # Non-maximum suppression - fjern overlappende detektioner
    final_matches = []
    while best_matches:
        best_match = best_matches.pop(0)
        final_matches.append(best_match)
        
        # Fjern overlappende matches
        non_overlapping_matches = []
        for match in best_matches:
            # Beregn centrum for begge matches
            best_center_x = best_match['position'][0] + best_match['template_size'][1] // 2
            best_center_y = best_match['position'][1] + best_match['template_size'][0] // 2
            
            match_center_x = match['position'][0] + match['template_size'][1] // 2
            match_center_y = match['position'][1] + match['template_size'][0] // 2
            
            # Beregn afstand mellem centrumspunkter
            distance = np.sqrt((best_center_x - match_center_x)**2 + (best_center_y - match_center_y)**2)
            
            # Hvis afstanden er større end et vist minimum, behold denne match
            if distance > min(best_match['template_size']) // 2:
                non_overlapping_matches.append(match)
        
        best_matches = non_overlapping_matches
    
    return len(final_matches), final_matches

def classify_tile(tile_image, terrain_model, templates, template_names, crown_threshold=0.6, visualize=False):
    """
    Klassificerer et tile ved hjælp af terrænmodel og kronedetektering.
    
    Args:
        tile_image: Billede af tile
        terrain_model: Indlæst terrænklassifikationsmodel
        templates: Liste af crown templates
        template_names: Liste af template navne
        crown_threshold: Threshold for kronedetektering
        visualize: Hvis True, vises en visualisering af resultatet
        
    Returns:
        Tuple med (klassificeret_terrain, antal_kroner, match_detaljer)
    """
    # Klassificer terræntype
    try:
        # Håndter forskellige formater af billedet
        if isinstance(tile_image, np.ndarray):
            # Konverter BGR til RGB hvis nødvendigt
            if tile_image.shape[2] == 3:  # Antager farvekanal som sidste dimension
                rgb_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = tile_image
        else:
            # Hvis ikke et numpy array, forsøg at konvertere
            rgb_image = np.array(tile_image)
        
        # Udtræk features til terrænklassifikation manuelt
        hsv_hist = extract_hsv_histogram(rgb_image)
        texture_hist = extract_texture_histogram(rgb_image)
        features = np.concatenate([hsv_hist, texture_hist])
        
        # Forudsig terræntype - reshape til 2D array
        features_2d = features.reshape(1, -1)
        terrain_type = terrain_model.predict_terrain(features_2d)[0]
    except Exception as e:
        print(f"Fejl under terrænklassifikation: {e}")
        terrain_type = "Unknown"
    
    # Detekter kroner
    try:
        crown_count, matches = detect_crowns_in_tile(
            tile_image, templates, template_names, crown_threshold
        )
    except Exception as e:
        print(f"Fejl under kronedetektering: {e}")
        crown_count = 0
        matches = []
    
    # Visualiser resultatet hvis ønsket
    if visualize and tile_image is not None:
        plt.figure(figsize=(10, 5))
        
        # Venstre: Originalt billede
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB))
        plt.title("Originalt tile")
        plt.axis('off')
        
        # Højre: Billede med klassifikation og kronedetektering
        plt.subplot(1, 2, 2)
        result_img = cv2.cvtColor(tile_image.copy(), cv2.COLOR_BGR2RGB)
        
        # Tegn rektangler omkring detekterede kroner
        for match in matches:
            x, y = match['position']
            w, h = match['template_size'][1], match['template_size'][0]
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        plt.imshow(result_img)
        plt.title(f"Klassificeret som: {terrain_type}, Kroner: {crown_count}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return terrain_type, crown_count, matches

def evaluate_on_sample(terrain_model, templates, template_names, samples_per_terrain=10, visualize=False):
    """
    Evaluerer det fulde system på et udvalg af tiles fra hver terræntype.
    
    Args:
        terrain_model: Indlæst terrænklassifikationsmodel
        templates: Liste af crown templates
        template_names: Liste af template navne
        samples_per_terrain: Antal samples at teste fra hver terræntype
        visualize: Hvis True, vises visualiseringer for nogle resultater
        
    Returns:
        Samlet statistik over performance
    """
    results = []
    actual_terrains = []
    predicted_terrains = []
    actual_crowns = []
    predicted_crowns = []
    
    # For hver terræntype
    for terrain_type in TERRAIN_TYPES:
        terrain_dir = os.path.join(TERRAIN_CATEGORIES_DIR, terrain_type)
        
        if not os.path.exists(terrain_dir):
            print(f"Advarsel: Mappen {terrain_dir} findes ikke.")
            continue
        
        # Find alle PNG-filer i mappen
        files = [f for f in os.listdir(terrain_dir) if f.endswith('.png')]
        
        # Vælg tilfældige samples hvis der er flere end samples_per_terrain
        if len(files) > samples_per_terrain:
            files = random.sample(files, samples_per_terrain)
        
        print(f"\nEvaluerer på {len(files)} samples fra {terrain_type}...")
        
        for i, file in enumerate(files):
            file_path = os.path.join(terrain_dir, file)
            
            # Indlæs tile
            tile = cv2.imread(file_path)
            if tile is None:
                print(f"Kunne ikke indlæse {file_path}")
                continue
            
            # Udtræk ground truth fra filnavnet
            actual_terrain, actual_crown_count = extract_info_from_filename(file)
            
            if actual_terrain is None:
                actual_terrain = terrain_type  # Brug mappenavnet hvis filnavnet ikke indeholder terræntype
            
            # Klassificer tile og detekter kroner
            predicted_terrain, predicted_crown_count, matches = classify_tile(
                tile, terrain_model, templates, template_names, 
                crown_threshold=0.6, visualize=(visualize and i == 0)  # Visualiser første tile fra hver type
            )
            
            # Gem resultater for confusion matrix
            actual_terrains.append(actual_terrain)
            predicted_terrains.append(predicted_terrain)
            actual_crowns.append(actual_crown_count)
            predicted_crowns.append(predicted_crown_count)
            
            # Gem resultater
            results.append({
                'filename': file,
                'actual_terrain': actual_terrain,
                'actual_crowns': actual_crown_count,
                'predicted_terrain': predicted_terrain,
                'predicted_crowns': predicted_crown_count,
                'terrain_correct': actual_terrain == predicted_terrain,
                'crowns_correct': actual_crown_count == predicted_crown_count,
                'fully_correct': (actual_terrain == predicted_terrain) and (actual_crown_count == predicted_crown_count)
            })
            
            # Print resultat for denne tile
            print(f"File: {file}")
            print(f"  Actual: {actual_terrain}, {actual_crown_count} crowns")
            print(f"  Predicted: {predicted_terrain}, {predicted_crown_count} crowns")
            print(f"  Correct: {'✓' if results[-1]['fully_correct'] else '✗'}")
    
    # Beregn samlede statistikker
    total = len(results)
    terrain_correct = sum(1 for r in results if r['terrain_correct'])
    crowns_correct = sum(1 for r in results if r['crowns_correct'])
    fully_correct = sum(1 for r in results if r['fully_correct'])
    
    terrain_accuracy = terrain_correct / total if total > 0 else 0
    crowns_accuracy = crowns_correct / total if total > 0 else 0
    overall_accuracy = fully_correct / total if total > 0 else 0
    
    # Sammenfat resultater
    stats = {
        'total_samples': total,
        'terrain_accuracy': terrain_accuracy,
        'crowns_accuracy': crowns_accuracy,
        'overall_accuracy': overall_accuracy,
        'terrain_correct': terrain_correct,
        'crowns_correct': crowns_correct,
        'fully_correct': fully_correct,
        'actual_terrains': actual_terrains,
        'predicted_terrains': predicted_terrains,
        'actual_crowns': actual_crowns,
        'predicted_crowns': predicted_crowns
    }
    
    # Generer detaljeret rapport
    print("\n===== PERFORMANCE REPORT =====")
    print(f"Total samples: {total}")
    print(f"Terrain classification accuracy: {terrain_accuracy:.2f} ({terrain_correct}/{total})")
    print(f"Crown detection accuracy: {crowns_accuracy:.2f} ({crowns_correct}/{total})")
    print(f"Overall accuracy: {overall_accuracy:.2f} ({fully_correct}/{total})")
    
    # Terrain classification report
    print("\nTerrain Classification Report:")
    try:
        print(classification_report(actual_terrains, predicted_terrains))
    except Exception as e:
        print(f"Error generating classification report: {e}")
    
    # Crown detection analysis
    print("\nCrown Detection Accuracy by Actual Crown Count:")
    crown_counts = {}
    for r in results:
        count = r['actual_crowns']
        if count not in crown_counts:
            crown_counts[count] = {'total': 0, 'correct': 0}
        crown_counts[count]['total'] += 1
        if r['crowns_correct']:
            crown_counts[count]['correct'] += 1
    
    for count, stats in sorted(crown_counts.items()):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {count} crowns: {accuracy:.2f} ({stats['correct']}/{stats['total']})")
    
    return stats, results

def plot_confusion_matrix(actual, predicted, labels, title):
    """
    Plotter en confusion matrix.
    
    Args:
        actual: Liste med faktiske værdier
        predicted: Liste med forudsagte værdier
        labels: Liste med labels til akser
        title: Titel på plottet
    """
    cm = confusion_matrix(actual, predicted, labels=labels)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    
    # Skriv værdier i cellerne
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Faktisk')
    plt.xlabel('Forudsagt')
    plt.show()

def main():
    """
    Hovedfunktion til at køre evalueringen.
    """
    # Indlæs terrænklassifikationsmodel
    terrain_model = load_terrain_model()
    if terrain_model is None:
        print("Kunne ikke indlæse terrænklassifikationsmodel. Afslutter.")
        return
    
    # Indlæs crown templates
    templates, template_names = load_crown_templates()
    if len(templates) != len(template_names):
        print("Kunne ikke indlæse alle crown templates. Afslutter.")
        return
    
    # Evaluer det fulde system
    print("Evaluerer det samlede system på udvalgte samples...")
    stats, results = evaluate_on_sample(
        terrain_model, templates, template_names, 
        samples_per_terrain=10, visualize=True
    )
    
    # Udtræk lister fra resultaterne for at plotte confusion matrices
    actual_terrains = [r['actual_terrain'] for r in results]
    predicted_terrains = [r['predicted_terrain'] for r in results]
    actual_crowns = [r['actual_crowns'] for r in results]
    predicted_crowns = [r['predicted_crowns'] for r in results]
    
    # Plot confusion matrix for terrænklassifikation hvis muligt
    try:
        unique_terrains = sorted(list(set(actual_terrains)))
        if len(unique_terrains) > 1 and not all(pt == "Unknown" for pt in predicted_terrains):
            plot_confusion_matrix(
                actual_terrains, 
                predicted_terrains,
                unique_terrains,
                "Terrain Classification Confusion Matrix"
            )
    except Exception as e:
        print(f"Kunne ikke plotte terrain confusion matrix: {e}")
    
    # Plot confusion matrix for kronedetektering hvis muligt
    try:
        unique_crown_counts = sorted(list(set(actual_crowns)))
        if len(unique_crown_counts) > 1:
            plot_confusion_matrix(
                actual_crowns, 
                predicted_crowns,
                unique_crown_counts,
                "Crown Detection Confusion Matrix"
            )
    except Exception as e:
        print(f"Kunne ikke plotte crown confusion matrix: {e}")
        
    # Gem resultaterne til en fil
    try:
        crown_results = {}
        for count in sorted(set(actual_crowns)):
            correct_cases = sum(1 for i, ac in enumerate(actual_crowns) 
                              if ac == count and predicted_crowns[i] == count)
            total_cases = sum(1 for ac in actual_crowns if ac == count)
            crown_results[count] = {
                'correct': correct_cases,
                'total': total_cases,
                'accuracy': correct_cases / total_cases if total_cases > 0 else 0
            }
            
        print("\nCrown Detection Results:")
        for count, result in crown_results.items():
            print(f"  {count} crown(s): {result['accuracy']:.2f} ({result['correct']}/{result['total']})")
    except Exception as e:
        print(f"Fejl ved beregning af kroneresultater: {e}")

if __name__ == "__main__":
    main()