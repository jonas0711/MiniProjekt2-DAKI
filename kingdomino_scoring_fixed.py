# Importerer de nødvendige biblioteker
import pandas as pd
import numpy as np
import re
import os

# Funktion til at udtrække koordinater fra en tekst streng
def extract_coordinates(coord_text):
    """
    Udtrækker X og Y koordinater fra en tekststreng som 'Tile (X, Y)'
    
    Args:
        coord_text: Tekst med koordinater (f.eks. 'Tile (0, 0)')
    
    Returns:
        tuple: (X, Y) koordinater som integers
    """
    # Bruger regex til at finde tallene i parenteserne
    match = re.search(r'\((\d+),\s*(\d+)\)', coord_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

# Funktion til at udtrække terræn og kroner
def extract_terrain_crowns(terrain_text):
    """
    Udtrækker terræn type og antal kroner fra en tekststreng
    
    Args:
        terrain_text: Tekst med terræn (f.eks. 'Forest', 'Forest1', 'Swamp2')
    
    Returns:
        tuple: (terræn_type, antal_kroner)
    """
    # Hvis terrain_text er None eller tom, returner default værdier
    if terrain_text is None or not isinstance(terrain_text, str) or not terrain_text.strip():
        return ("Unknown", 0)
    
    # Finder terræn og antal kroner (hvis der er nogen)
    match = re.search(r'(\D+)(\d*)', terrain_text)
    if match:
        terrain = match.group(1).strip()
        crowns = match.group(2)
        # Hvis der er et tal efter terrænnavn, er det antal kroner, ellers 0
        crown_count = int(crowns) if crowns else 0
        return (terrain, crown_count)
    return (terrain_text, 0)

# Funktion til at finde sammenhængende områder af samme terræn
def find_connected_territories(board, x, y, terrain, visited, home_coords):
    """
    Finder alle sammenhængende tiles af samme terræn type ved hjælp af dybde-først søgning (DFS)
    
    Args:
        board: Bræt med terrain information
        x, y: Koordinater for den aktuelle tile
        terrain: Terræn type vi leder efter
        visited: Set af allerede besøgte koordinater
        home_coords: Koordinater for Home tile (startbrikken med slottet)
    
    Returns:
        list: Liste af koordinater for sammenhængende tiles
    """
    # Hvis koordinaterne er uden for brættet (5x5 grid) eller allerede besøgt, returner tom liste
    if x < 0 or x >= 5 or y < 0 or y >= 5:
        return []
    
    # Hvis koordinaterne allerede er besøgt, returner tom liste
    if (x, y) in visited:
        return []
    
    # Tjek om koordinaterne findes på brættet
    if (x, y) not in board:
        return []
    
    # Home tile (startbrikken) giver ikke point selv, men den kan forbinde territorier
    # Ifølge reglerne skal Home tile ikke tælles med i territoriet selv
    if (x, y) == home_coords:
        # Marker Home tile som besøgt
        visited.add((x, y))
        
        # Undersøg de tilstødende positioner (op, ned, venstre, højre)
        territory = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            territory.extend(find_connected_territories(board, x + dx, y + dy, terrain, visited, home_coords))
        
        return territory
    # Hvis terrænet på denne position ikke matcher det vi leder efter, returner tom liste
    elif board[(x, y)][0] != terrain:
        return []
    
    # Marker denne position som besøgt
    visited.add((x, y))
    
    # Tilføj denne position til vores resultat liste
    territory = [(x, y)]
    
    # Undersøg de tilstødende positioner (op, ned, venstre, højre)
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        territory.extend(find_connected_territories(board, x + dx, y + dy, terrain, visited, home_coords))
    
    return territory

# Funktion til at beregne scoren for en territorium
def calculate_territory_score(board, territory):
    """
    Beregner scoren for et territorium
    
    Args:
        board: Dictionary med tile information
        territory: Liste af koordinater for tiles i territoriet
    
    Returns:
        int: Score for territoriet (antal tiles * antal kroner)
    """
    num_tiles = len(territory)
    total_crowns = sum(board[tile][1] for tile in territory)
    
    # Ifølge reglerne giver territorier uden kroner 0 point
    if total_crowns == 0:
        return 0
    
    # Score er antal tiles ganget med antal kroner
    return num_tiles * total_crowns

# Funktion til at beregne samlet score for et bræt
def calculate_board_score(board_data):
    """
    Beregner den samlede score for et bræt baseret på spillets regler
    
    Args:
        board_data: DataFrame med brætdata
    
    Returns:
        tuple: (Samlet score for brættet, detaljer om territorier)
    """
    # Opret et dictionary til at gemme board information
    board = {}
    home_coords = None
    
    # Gennemgå alle rækker i dataframen for at opbygge brættet
    for _, row in board_data.iterrows():
        coord_text = row.iloc[0]
        terrain_text = row.iloc[1]
        
        # Udtrækker koordinater, terræn og kroner
        coords = extract_coordinates(coord_text)
        terrain, crowns = extract_terrain_crowns(terrain_text)
        
        if coords:
            # Tilføj information til board dictionary
            board[coords] = (terrain, crowns)
            
            # Gem koordinater for Home tile (slottet)
            if terrain == 'Home':
                home_coords = coords
    
    # Hvis vi ikke fandt nogen Home tile, så antag det er i midten (2,2)
    if home_coords is None:
        home_coords = (2, 2)
        print("Advarsel: Ingen Home tile fundet. Antager position (2,2).")
    
    # Set til at holde styr på besøgte tiles
    visited = set()
    
    # Liste til at gemme alle territorier
    territories = []
    
    # Først markerer vi Home tile som besøgt, da den ikke skal være en del af nogen territorier
    if home_coords in board:
        visited.add(home_coords)
    
    # Gennemgå alle tiles på brættet for at finde territorier
    for (x, y) in board:
        if (x, y) not in visited:
            terrain = board[(x, y)][0]
            
            # Spring over non-scoring terrain typer
            if terrain in ['Home', 'Table', 'Unknown']:
                visited.add((x, y))
                continue
                
            # Find sammenhængende territorium
            territory = find_connected_territories(board, x, y, terrain, visited, home_coords)
            if territory:
                territories.append((terrain, territory))
    
    # Beregn score for hvert territorium og læg dem sammen
    total_score = 0
    territory_details = []
    
    for terrain_type, territory in territories:
        # Beregn detaljer for territoriet
        crowns = sum(board[tile][1] for tile in territory)
        tile_count = len(territory)
        score = calculate_territory_score(board, territory)
        
        # Tilføj til den samlede score
        total_score += score
        
        # Gem detaljer om territoriet
        territory_details.append({
            'terrain_type': terrain_type,
            'tiles': tile_count,
            'crowns': crowns,
            'score': score
        })
    
    return total_score, territory_details

# Funktion til at beregne bonus for perfekt 5x5 grid (Harmony variant)
def calculate_harmony_bonus(board_data):
    """
    Beregner om brættet fortjener en Harmony bonus (5 point for et komplet 5x5 grid uden huller)
    
    Args:
        board_data: DataFrame med brætdata
    
    Returns:
        int: 5 hvis der er Harmony bonus, ellers 0
    """
    # Opret et set til at gemme alle koordinater på brættet
    board_coords = set()
    
    # Gennemgå alle rækker i dataframen
    for _, row in board_data.iterrows():
        coord_text = row.iloc[0]
        coords = extract_coordinates(coord_text)
        if coords:
            board_coords.add(coords)
    
    # Tjek om der er præcis 25 unikke koordinater (et komplet 5x5 grid)
    if len(board_coords) == 25:
        # Tjek om alle koordinater inden for 5x5 grid er dækket
        all_coords = {(x, y) for x in range(5) for y in range(5)}
        if board_coords == all_coords:
            return 5
    
    return 0

# Hovedfunktion
def main():
    """
    Hovedfunktion der indlæser Excel-filen og beregner scores for alle brætter
    """
    # Stien til Excel-filen med korrigerede terrænnavne
    excel_file = 'kingdomino_labels_fixed.xlsx'
    
    # Tjek om filen eksisterer
    if not os.path.exists(excel_file):
        print(f"Fejl: Filen {excel_file} blev ikke fundet.")
        return
    
    # Indlæser Excel med alle faner
    print(f"Indlæser Excel-filen {excel_file}...")
    xl = pd.ExcelFile(excel_file)
    
    # Dictionary til at gemme alle scores
    all_scores = {}
    
    # For hver fane (bræt) i Excel-filen
    for sheet_name in xl.sheet_names:
        # Spring over Sheet fanen hvis den ikke indeholder brætdata
        if sheet_name == 'Sheet':
            continue
            
        print(f"Beregner score for bræt {sheet_name}...")
        
        try:
            # Indlæs data for dette bræt
            board_data = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Tjek om vi har adgang til kolonne B
            if board_data.shape[1] < 2:
                print(f"Advarsel: Fane {sheet_name} har ikke kolonne B. Springer over.")
                continue
            
            # Beregn score for dette bræt
            score, territory_details = calculate_board_score(board_data)
            
            # Beregn bonus for Harmony variant (perfekt 5x5 grid)
            harmony_bonus = calculate_harmony_bonus(board_data)
            total_score = score + harmony_bonus
            
            # Gem scoren i vores dictionary
            all_scores[sheet_name] = {
                'base_score': score,
                'harmony_bonus': harmony_bonus,
                'total_score': total_score,
                'territory_details': territory_details
            }
            
        except Exception as e:
            print(f"Fejl ved beregning af score for bræt {sheet_name}: {e}")
    
    # Opret en DataFrame med alle scores
    scores_df = pd.DataFrame({
        'Board': list(all_scores.keys()),
        'Base Score': [all_scores[board]['base_score'] for board in all_scores],
        'Harmony Bonus': [all_scores[board]['harmony_bonus'] for board in all_scores],
        'Total Score': [all_scores[board]['total_score'] for board in all_scores]
    })
    
    # Sortér efter Total Score (højest først)
    scores_df = scores_df.sort_values('Total Score', ascending=False)
    
    # Gem resultater til en Excel-fil
    output_file = 'kingdomino_final_scores_fixed.xlsx'
    print(f"\nGemmer resultater til {output_file}...")
    
    # Gem både summary og detaljerede ark
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        scores_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Opret et detaljeret ark med information om hvert bræt
        for board in all_scores:
            details = all_scores[board]['territory_details']
            if details:
                details_df = pd.DataFrame(details)
                details_df = details_df.sort_values('score', ascending=False)
                details_df.to_excel(writer, sheet_name=f'Board {board}', index=False)
    
    print(f"Resultater gemt til {output_file}")
    
    # Print top 5 brætter med højeste score
    print("\nTop 5 brætter med højeste score:")
    top_5_boards = scores_df.head(5)
    for _, row in top_5_boards.iterrows():
        board = row['Board']
        base_score = row['Base Score']
        harmony_bonus = row['Harmony Bonus']
        total_score = row['Total Score']
        
        bonus_text = f" (inkl. {harmony_bonus} bonus point for harmoni)" if harmony_bonus > 0 else ""
        print(f"Bræt {board}: {total_score} point{bonus_text}")
    
    # Print detaljeret information for top 5 brætter
    print("\nDetaljer for top 5 brætter:")
    for board in top_5_boards['Board']:
        base_score = all_scores[board]['base_score']
        harmony_bonus = all_scores[board]['harmony_bonus']
        total_score = all_scores[board]['total_score']
        
        bonus_text = f" (inkl. {harmony_bonus} bonus point for harmoni)" if harmony_bonus > 0 else ""
        print(f"\nBræt {board} (Total score: {total_score} point{bonus_text})")
        
        if all_scores[board]['territory_details']:
            print("Territorier:")
            for t in sorted(all_scores[board]['territory_details'], key=lambda x: x['score'], reverse=True):
                crown_text = "krone" if t['crowns'] == 1 else "kroner"
                tile_text = "felter" if t['tiles'] != 1 else "felt"
                
                print(f"  - {t['terrain_type']}: {t['tiles']} {tile_text}, {t['crowns']} {crown_text} = {t['score']} point")
        else:
            print("Ingen territorier med kroner fundet.")

# Kør hovedfunktionen hvis dette script køres direkte
if __name__ == "__main__":
    main() 