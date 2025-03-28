import numpy as np
from collections import deque

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