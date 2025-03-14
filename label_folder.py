import json
import os
import shutil

def organize_tiles_by_terrain(labels_file, tiles_dir, output_base_dir):
    """
    Organiserer alle tile-billeder i undermapper baseret på terræntype
    
    Args:
        labels_file: Sti til JSON-filen med labels
        tiles_dir: Sti til mappen med alle tile-billeder
        output_base_dir: Basismappe, hvor terræn-undermapper skal oprettes
    """
    # Indlæs labels-data fra JSON-filen
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # Opret en dictionary til at gemme alle unikke terræntyper
    terrain_types = set()
    
    # Find alle unikke terræntyper
    for board in labels_data.values():
        for tile_info in board.values():
            terrain = tile_info["terrain"]
            terrain_types.add(terrain)
    
    # Opret basismappe, hvis den ikke findes
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Opret undermapper for hver terræntype
    for terrain in terrain_types:
        terrain_dir = os.path.join(output_base_dir, terrain)
        if not os.path.exists(terrain_dir):
            os.makedirs(terrain_dir)
            print(f"Oprettet mappe: {terrain_dir}")
    
    # Flyt billeder til de korrekte mapper
    copied_count = 0
    for board_id, board_data in labels_data.items():
        for tile_pos, tile_info in board_data.items():
            filename = tile_info["filename"]
            terrain = tile_info["terrain"]
            crowns = tile_info["crowns"]
            
            # Definer kilde- og målstier
            source_path = os.path.join(tiles_dir, filename)
            
            # Tilføj kroner til filnavnet for at gøre det mere informativt
            new_filename = f"{board_id}_tile_{tile_pos}_{terrain}_{crowns}crowns.png"
            target_path = os.path.join(output_base_dir, terrain, new_filename)
            
            # Kopier filen, hvis den findes
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                copied_count += 1
                
                # Print status for hver 100. fil
                if copied_count % 100 == 0:
                    print(f"Kopieret {copied_count} filer...")
            else:
                print(f"Advarsel: Filen {source_path} findes ikke")
    
    print(f"\nAfsluttet! Organiseret {copied_count} billeder i {len(terrain_types)} terrænmapper.")
    print("Terræntyper: " + ", ".join(sorted(terrain_types)))

# Stier til brug
labels_file = "tile_labels_mapping.json"  # Erstat med den faktiske sti
tiles_dir = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"  # Erstat med den faktiske sti
output_base_dir = "KingDominoDataset/TerrainCategories"  # Basis outputmappe

# Kør funktionen
organize_tiles_by_terrain(labels_file, tiles_dir, output_base_dir)