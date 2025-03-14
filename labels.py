import pandas as pd
import os
import re
import json
from collections import defaultdict

def extract_coordinates(coord_text):
    """
    Udtrækker X og Y koordinater fra en tekststreng som 'Tile (X, Y)'
    
    Args:
        coord_text: Tekst med koordinater (fx 'Tile (0, 0)')
    
    Returns:
        tuple: (X, Y) koordinater som integers, eller None hvis ikke fundet
    """
    # Brug regex til at finde tal i parentes
    match = re.search(r'\((\d+),\s*(\d+)\)', coord_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def extract_terrain_and_crowns(terrain_text):
    """
    Udtrækker terræntype og antal kroner fra en tekststreng
    
    Args:
        terrain_text: Tekst med terræntype (fx 'Forest', 'Forest1', 'Swamp2')
    
    Returns:
        tuple: (terræntype, antal_kroner)
    """
    # Hvis terrain_text er None eller tom, returner default værdier
    if terrain_text is None or not isinstance(terrain_text, str) or not terrain_text.strip():
        return ("Unknown", 0)
    
    # Finder terræntype og antal kroner (hvis der er nogen)
    match = re.search(r'(\D+)(\d*)', terrain_text)
    if match:
        terrain = match.group(1).strip()
        crowns = match.group(2)
        # Hvis der er et tal efter terræntype, er det antal kroner, ellers 0
        crown_count = int(crowns) if crowns else 0
        return (terrain, crown_count)
    return (terrain_text, 0)

def load_labels_from_excel(excel_file):
    """
    Indlæser terrænlabels fra en Excel-fil
    
    Args:
        excel_file: Sti til Excel-filen
    
    Returns:
        dict: Dictionary med { 'board_name': { (x, y): (terrain, crowns) } }
    """
    # Tjek om filen eksisterer
    if not os.path.exists(excel_file):
        print(f"Fejl: Filen {excel_file} blev ikke fundet.")
        return {}
    
    # Opret en dictionary til at gemme alle labels
    all_labels = {}
    
    # Indlæs Excel-filen
    xl = pd.ExcelFile(excel_file)
    
    # Gennemgå hvert ark (hvert ark repræsenterer en spilleplade)
    for sheet_name in xl.sheet_names:
        # Spring over Sheet fanen hvis den ikke indeholder brætdata
        if sheet_name == 'Sheet':
            continue
        
        # Indlæs data for dette bræt
        board_data = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Opret en dictionary til at gemme labels for dette bræt
        board_labels = {}
        
        # Gennemløb hver række i dataframen
        for _, row in board_data.iterrows():
            # Hent koordinattekst og terræntekst
            if len(row) >= 2:  # Sikrer, at der er mindst 2 kolonner
                coord_text = row.iloc[0]
                terrain_text = row.iloc[1]
                
                # Udtrækker koordinater
                coords = extract_coordinates(coord_text)
                
                # Udtrækker terræntype og antal kroner
                if coords:
                    terrain, crowns = extract_terrain_and_crowns(terrain_text)
                    board_labels[coords] = (terrain, crowns)
        
        # Tilføj dette bræts labels til den samlede dictionary
        all_labels[sheet_name] = board_labels
    
    return all_labels

def map_labels_to_extracted_tiles(labels_dict, tiles_dir, output_file):
    """
    Mapper labels til extraherede tiles og gemmer resultatet som JSON
    
    Args:
        labels_dict: Dictionary med labels for hvert bræt
        tiles_dir: Mappe med ekstraherede tiles
        output_file: Fil, hvor mappingen skal gemmes
    """
    # Tjek om mappen eksisterer
    if not os.path.exists(tiles_dir):
        print(f"Fejl: Mappen {tiles_dir} blev ikke fundet.")
        return
    
    # Opret en dictionary til at gemme mappingen
    tile_mapping = defaultdict(dict)
    
    # Få en liste over alle filer i mappen
    tiles_files = os.listdir(tiles_dir)
    
    # Gennemløb hver fil
    for filename in tiles_files:
        # Split filnavnet for at få information
        parts = filename.split('_')
        
        # Tjek om filnavnet har det forventede format
        if len(parts) >= 4 and parts[1] == "tile":
            board_name = parts[0]
            row = int(parts[2])
            col = int(parts[3].split('.')[0])
            
            # Tjek om vi har labels for dette bræt
            if board_name in labels_dict:
                board_labels = labels_dict[board_name]
                
                # Find label for denne position (hvis den findes)
                coords = (col, row)  # Bemærk: Koordinaterne er (x, y), dvs. (col, row)
                
                if coords in board_labels:
                    terrain, crowns = board_labels[coords]
                    
                    # Tilføj til mappingen
                    tile_mapping[board_name][f"{row}_{col}"] = {
                        "filename": filename,
                        "terrain": terrain,
                        "crowns": crowns
                    }
                else:
                    print(f"Advarsel: Ingen label fundet for {board_name} tile ({row}, {col})")
    
    # Gem mappingen som JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tile_mapping, f, indent=2)
    
    print(f"Mapping gemt til {output_file}")
    
    return tile_mapping

def create_labeled_dataset(excel_file, tiles_dir, output_file='tile_labels_mapping.json'):
    """
    Skaber et mappet datasæt ved at koble Excel-labels til ekstraherede tiles
    
    Args:
        excel_file: Sti til Excel-filen med labels
        tiles_dir: Mappe med ekstraherede tiles
        output_file: Fil, hvor mappingen skal gemmes
    
    Returns:
        dict: Dictionary med den gemte mapping
    """
    # Indlæs labels fra Excel
    labels_dict = load_labels_from_excel(excel_file)
    
    # Map labels til ekstraherede tiles
    mapping = map_labels_to_extracted_tiles(labels_dict, tiles_dir, output_file)
    
    # Vis en opsummering
    print(f"Labels indlæst for {len(labels_dict)} brætter.")
    
    # Vis nogle statistikker
    terrain_counts = defaultdict(int)
    crown_counts = defaultdict(int)
    
    for board_name, tiles in mapping.items():
        for tile_pos, tile_info in tiles.items():
            terrain = tile_info["terrain"]
            crowns = tile_info["crowns"]
            
            terrain_counts[terrain] += 1
            crown_counts[crowns] += 1
    
    print("\nTerrænfordelingsstatistik:")
    for terrain, count in sorted(terrain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {terrain}: {count} tiles")
    
    print("\nKronefordelingsstatistik:")
    for crowns, count in sorted(crown_counts.items()):
        print(f"  {crowns} kroner: {count} tiles")
    
    return mapping

# Eksempel på brug
if __name__ == "__main__":
    # Definer stier
    excel_file = "kingdomino_labels_fixed.xlsx"
    tiles_dir = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
    output_file = "tile_labels_mapping.json"
    
    # Opret det mappede datasæt
    mapping = create_labeled_dataset(excel_file, tiles_dir, output_file)