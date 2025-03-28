import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import re

def load_template(template_path):
    """
    Indlæser et template billede og konverterer det til gråskala.
    
    Args:
        template_path: Sti til template billedet
        
    Returns:
        Det indlæste og preprocessede template billede
    """
    # Indlæs template
    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Kunne ikke indlæse template fra {template_path}")
    
    # Konverter til gråskala for at reducere effekten af farveforskelle
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    return template_gray

def detect_crowns_in_tile(tile_image, templates, template_names, threshold=0.6, visualize=False):
    """
    Detekterer kroner i et tile ved hjælp af template matching.
    
    Args:
        tile_image: Billedet af feltet der skal undersøges
        templates: Liste af templates (kroner i forskellige orienteringer)
        template_names: Liste af tilsvarende template navne
        threshold: Minimum værdi for at acceptere et match (0.0-1.0)
        visualize: Hvis True, vises resultatet visuelt
    
    Returns:
        Antal detekterede kroner, bedste match-værdier og positioner
    """
    # Konverter tile til gråskala for template matching
    tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
    tile_gray = cv2.cvtColor(tile_gray, cv2.COLOR_RGB2GRAY)
    
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
    
    # Visualiser resultaterne hvis ønsket
    if visualize and final_matches:
        plt.figure(figsize=(10, 5))
        
        # Venstre: Originalt billede
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB))
        plt.title("Originalt tile")
        plt.axis('off')
        
        # Højre: Billede med markerede detektioner
        plt.subplot(1, 2, 2)
        result_img = cv2.cvtColor(tile_image.copy(), cv2.COLOR_BGR2RGB)
        
        for match in final_matches:
            # Tegn rektangel omkring match
            x, y = match['position']
            w, h = match['template_size'][1], match['template_size'][0]
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Tilføj match værdi og template navn
            text = f"{match['template']} ({match['value']:.2f})"
            cv2.putText(result_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        plt.imshow(result_img)
        plt.title(f"Detekterede kroner: {len(final_matches)}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return len(final_matches), final_matches

def extract_crown_count_from_filename(filename):
    """
    Udtrækker det faktiske antal kroner fra filnavnet.
    
    Args:
        filename: Filnavn at analysere
        
    Returns:
        Antal kroner fra filnavnet eller 0 hvis ingen information findes
    """
    # Regex til at finde crowns_X i filnavnet
    match = re.search(r'(\d+)crowns', filename)
    if match:
        return int(match.group(1))
    return 0

def evaluate_crown_detection(terrain_dir, templates, template_names, threshold=0.6, max_files=None, visualize=False):
    """
    Evaluerer kronedetektionsalgoritmen på tiles fra en bestemt terræntype.
    
    Args:
        terrain_dir: Sti til mappen med tiles for en terræntype
        templates: Liste af template billeder
        template_names: Liste af template navne
        threshold: Minimum værdi for at acceptere et match
        max_files: Maksimalt antal filer at teste (None = alle)
        visualize: Hvis True, visualiseres nogle resultater
    
    Returns:
        Statistik over korrekthed (præcision, recall, osv.)
    """
    # Find alle PNG-filer i mappen
    files = [f for f in os.listdir(terrain_dir) if f.endswith('.png')]
    
    if max_files is not None:
        files = files[:max_files]
    
    results = []
    for i, file in enumerate(files):
        file_path = os.path.join(terrain_dir, file)
        
        # Indlæs tile
        tile = cv2.imread(file_path)
        if tile is None:
            print(f"Kunne ikke indlæse {file_path}")
            continue
        
        # Hent faktisk antal kroner fra filnavnet
        actual_crowns = extract_crown_count_from_filename(file)
        
        # Detekter kroner
        detected_crowns, matches = detect_crowns_in_tile(
            tile, templates, template_names, threshold, 
            visualize=(visualize and i < 5)  # Vis kun de første 5 resultater
        )
        
        # Gem resultater
        results.append({
            'filename': file,
            'actual_crowns': actual_crowns,
            'detected_crowns': detected_crowns,
            'correct': actual_crowns == detected_crowns
        })
        
        # Udskriv løbende status
        if (i+1) % 10 == 0:
            print(f"Evalueret {i+1}/{len(files)} filer...")
    
    # Beregn statistik
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total if total > 0 else 0
    
    # Beregn precision, recall og F1-score
    true_positives = sum(1 for r in results if r['actual_crowns'] > 0 and r['detected_crowns'] > 0)
    false_positives = sum(1 for r in results if r['actual_crowns'] == 0 and r['detected_crowns'] > 0)
    false_negatives = sum(1 for r in results if r['actual_crowns'] > 0 and r['detected_crowns'] == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats = {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Vis detaljeret oversigt over fejl
    print("\nEksempler på fejldetektioner:")
    errors = [r for r in results if not r['correct']]
    for i, error in enumerate(errors[:5]):  # Vis kun de første 5 fejl
        print(f"Fil: {error['filename']}, Faktisk: {error['actual_crowns']}, Detekteret: {error['detected_crowns']}")
    
    if len(errors) > 5:
        print(f"...og {len(errors) - 5} mere.")
    
    return stats, results

def main():
    """
    Hovedfunktion til at køre kronedetektering og evaluering.
    """
    # Stier til templates
    template_dir = "KingDominoDataset/Crown_Templates"
    template_paths = [
        os.path.join(template_dir, "Crown_up.png"),
        os.path.join(template_dir, "Crown_down.png"),
        os.path.join(template_dir, "Crown_left.png"),
        os.path.join(template_dir, "Crown_right.png")
    ]
    template_names = ["Up", "Down", "Left", "Right"]
    
    # Indlæs templates
    print("Indlæser templates...")
    templates = []
    for path in template_paths:
        try:
            template = load_template(path)
            templates.append(template)
            print(f"Indlæst template: {path}, Form: {template.shape}")
        except FileNotFoundError as e:
            print(f"Advarsel: {e}")
    
    # Hvis nogle templates ikke kunne indlæses, stop programmet
    if len(templates) != len(template_paths):
        print("Kunne ikke indlæse alle templates. Kontroller filstierne.")
        return
    
    # Stier til terrænmapper
    terrain_categories_dir = "KingDominoDataset/TerrainCategories"
    terrain_types = ["Field", "Forest", "Grassland", "Lake", "Mine", "Swamp"]
    
    # Evalueringsparametre
    threshold = 0.6  # Template matching threshold
    max_files = 50   # Maksimalt antal filer at teste pr. terræntype
    
    # Evaluer for hver terræntype
    all_results = {}
    for terrain in terrain_types:
        terrain_dir = os.path.join(terrain_categories_dir, terrain)
        
        if not os.path.exists(terrain_dir):
            print(f"Advarsel: Mappen {terrain_dir} findes ikke.")
            continue
        
        print(f"\nEvaluerer kronedetektering for {terrain}...")
        stats, results = evaluate_crown_detection(
            terrain_dir, templates, template_names, 
            threshold, max_files, visualize=True
        )
        
        print(f"Resultater for {terrain}:")
        print(f"  Nøjagtighed: {stats['accuracy']:.2f}")
        print(f"  Precision: {stats['precision']:.2f}")
        print(f"  Recall: {stats['recall']:.2f}")
        print(f"  F1-score: {stats['f1_score']:.2f}")
        
        all_results[terrain] = {
            'stats': stats,
            'results': results
        }
    
    # Beregn samlet statistik på tværs af alle terræntyper
    total_correct = sum(r['stats']['correct'] for r in all_results.values())
    total_files = sum(r['stats']['total'] for r in all_results.values())
    overall_accuracy = total_correct / total_files if total_files > 0 else 0
    
    print("\nSamlet statistik på tværs af alle terræntyper:")
    print(f"  Total antal tiles evalueret: {total_files}")
    print(f"  Korrekte detektioner: {total_correct}")
    print(f"  Samlet nøjagtighed: {overall_accuracy:.2f}")
    
    # Eksperimenter med forskellige threshold-værdier
    if False:  # Sæt til True for at køre dette eksperiment
        print("\nEksperimenterer med forskellige threshold-værdier...")
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        accuracies = []
        
        for thresh in thresholds:
            print(f"\nTester threshold = {thresh}")
            
            # Vælg en terræntype til test
            terrain = "Forest"  # Eksempel
            terrain_dir = os.path.join(terrain_categories_dir, terrain)
            
            stats, _ = evaluate_crown_detection(
                terrain_dir, templates, template_names, 
                thresh, max_files=20, visualize=False
            )
            
            accuracies.append(stats['accuracy'])
            print(f"  Nøjagtighed ved threshold {thresh}: {stats['accuracy']:.2f}")
        
        # Plot resultaterne
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, accuracies, 'o-')
        plt.xlabel('Threshold værdi')
        plt.ylabel('Nøjagtighed')
        plt.title('Nøjagtighed vs. Threshold værdi')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()