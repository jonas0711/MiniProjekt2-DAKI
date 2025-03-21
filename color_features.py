import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json

def compute_rgb_statistics(image):
    """
    Beregner statistiske mål for RGB-komponenterne i et billede.
    
    Args:
        image: RGB-billede (numpy array)
    
    Returns:
        dict: Dictionary med statistiske mål for hver farvekanal (R, G, B)
    """
    # Adskil RGB-kanaler
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    
    # Beregn statistiske mål
    stats = {
        'R': {
            'mean': np.mean(r_channel),
            'median': np.median(r_channel),
            'std': np.std(r_channel),
            'min': np.min(r_channel),
            'max': np.max(r_channel)
        },
        'G': {
            'mean': np.mean(g_channel),
            'median': np.median(g_channel),
            'std': np.std(g_channel),
            'min': np.min(g_channel),
            'max': np.max(g_channel)
        },
        'B': {
            'mean': np.mean(b_channel),
            'median': np.median(b_channel),
            'std': np.std(b_channel),
            'min': np.min(b_channel),
            'max': np.max(b_channel)
        }
    }
    
    return stats

def convert_to_hsv(image):
    """
    Konverterer et RGB-billede til HSV-farverum.
    
    Args:
        image: RGB-billede (numpy array)
    
    Returns:
        numpy array: HSV-billede
    """
    # OpenCV forventer BGR-format, så vi skal konvertere fra RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Billedet skal være i RGB-format")
    
    return hsv_image

def compute_hsv_statistics(image):
    """
    Beregner statistiske mål for HSV-komponenterne i et billede.
    
    Args:
        image: RGB-billede (numpy array)
    
    Returns:
        dict: Dictionary med statistiske mål for hver HSV-komponent (H, S, V)
    """
    # Konverter til HSV
    hsv_image = convert_to_hsv(image)
    
    # Adskil HSV-kanaler
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]
    
    # Beregn statistiske mål
    stats = {
        'H': {
            'mean': np.mean(h_channel),
            'median': np.median(h_channel),
            'std': np.std(h_channel),
            'min': np.min(h_channel),
            'max': np.max(h_channel)
        },
        'S': {
            'mean': np.mean(s_channel),
            'median': np.median(s_channel),
            'std': np.std(s_channel),
            'min': np.min(s_channel),
            'max': np.max(s_channel)
        },
        'V': {
            'mean': np.mean(v_channel),
            'median': np.median(v_channel),
            'std': np.std(v_channel),
            'min': np.min(v_channel),
            'max': np.max(v_channel)
        }
    }
    
    return stats

def compute_color_histogram(image, color_space="rgb", bins=32, normalize=True):
    """
    Beregner farvehistogram for et billede.
    
    Args:
        image: RGB-billede (numpy array)
        color_space: Farverum at beregne histogram for ("rgb" eller "hsv")
        bins: Antal bins i histogrammet
        normalize: Om histogrammet skal normaliseres
    
    Returns:
        tuple: (histogrammer for hver kanal, bin-kanter, kanal-navne)
    """
    if color_space.lower() == "hsv":
        # Konverter til HSV
        processed_image = convert_to_hsv(image)
        # Definer ranges for HSV (OpenCV HSV: H: 0-180, S: 0-255, V: 0-255)
        ranges = [(0, 180), (0, 255), (0, 255)]
        channel_names = ['H', 'S', 'V']
    else:  # Default to RGB
        processed_image = image.copy()
        ranges = [(0, 255), (0, 255), (0, 255)]
        channel_names = ['R', 'G', 'B']
    
    histograms = []
    bin_edges = []
    
    # Beregn histogram for hver kanal
    for i in range(3):
        hist, edges = np.histogram(
            processed_image[:, :, i].flatten(), 
            bins=bins, 
            range=ranges[i], 
            density=normalize
        )
        histograms.append(hist)
        bin_edges.append(edges)
    
    return histograms, bin_edges, channel_names

def compute_patch_histograms(image, patch_size=(4, 4), color_space="rgb", bins=8):
    """
    Beregner farvehistogrammer for lokale områder (patches) i et billede.
    
    Args:
        image: RGB-billede (numpy array)
        patch_size: Størrelse af hver patch (height, width)
        color_space: Farverum at beregne histogram for ("rgb" eller "hsv")
        bins: Antal bins i histogrammet
    
    Returns:
        numpy array: Array af histogram-features for hver patch
    """
    # Konvertér billedet til det ønskede farverum
    if color_space.lower() == "hsv":
        processed_image = convert_to_hsv(image)
        # Definer ranges for HSV (OpenCV HSV: H: 0-180, S: 0-255, V: 0-255)
        ranges = [(0, 180), (0, 255), (0, 255)]
    else:  # Default to RGB
        processed_image = image.copy()
        ranges = [(0, 255), (0, 255), (0, 255)]
    
    height, width, _ = processed_image.shape
    
    # Beregn antal patches i højden og bredden
    num_patches_h = height // patch_size[0]
    num_patches_w = width // patch_size[1]
    
    # Initialisér array til patch-features
    all_patch_features = []
    
    # Gennemløb alle patches
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Udtræk patch
            h_start = i * patch_size[0]
            h_end = h_start + patch_size[0]
            w_start = j * patch_size[1]
            w_end = w_start + patch_size[1]
            
            patch = processed_image[h_start:h_end, w_start:w_end, :]
            
            # Beregn histogram for hver kanal
            patch_features = []
            for k in range(3):
                hist, _ = np.histogram(
                    patch[:, :, k].flatten(), 
                    bins=bins, 
                    range=ranges[k],
                    density=True
                )
                patch_features.extend(hist)
            
            all_patch_features.append(patch_features)
    
    return np.array(all_patch_features)

def visualize_color_analysis(image, title="Farveanalyse"):
    """
    Visualiserer farveanalyse for et billede med RGB/HSV statistik og histogrammer.
    
    Args:
        image: RGB-billede (numpy array)
        title: Titel på visualiseringen
        
    Returns:
        fig: Matplotlib figure objekt
    """
    # Beregn statistik og histogrammer
    rgb_stats = compute_rgb_statistics(image)
    hsv_stats = compute_hsv_statistics(image)
    rgb_hist, rgb_edges, rgb_names = compute_color_histogram(image, "rgb")
    hsv_hist, hsv_edges, hsv_names = compute_color_histogram(image, "hsv")
    
    # Opret figur med subplots
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=16)
    
    # Vis originalbillede
    ax1 = fig.add_subplot(231)
    ax1.imshow(image)
    ax1.set_title("Original")
    ax1.axis('off')
    
    # Vis HSV-billede
    ax2 = fig.add_subplot(232)
    hsv_image = convert_to_hsv(image)
    # Konverter HSV tilbage til RGB for visning
    hsv_for_display = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    hsv_for_display = cv2.cvtColor(hsv_for_display, cv2.COLOR_BGR2RGB)
    ax2.imshow(hsv_for_display)
    ax2.set_title("HSV")
    ax2.axis('off')
    
    # Vis RGB-histogrammer
    ax3 = fig.add_subplot(233)
    for i, (hist, name) in enumerate(zip(rgb_hist, rgb_names)):
        bin_centers = (rgb_edges[i][:-1] + rgb_edges[i][1:]) / 2
        ax3.plot(bin_centers, hist, color='rgb'[i], label=name)
    ax3.set_title("RGB Histogram")
    ax3.legend()
    
    # Vis HSV-histogrammer
    ax4 = fig.add_subplot(234)
    for i, (hist, name) in enumerate(zip(hsv_hist, hsv_names)):
        bin_centers = (hsv_edges[i][:-1] + hsv_edges[i][1:]) / 2
        ax4.plot(bin_centers, hist, label=name)
    ax4.set_title("HSV Histogram")
    ax4.legend()
    
    # Vis RGB statistik
    ax5 = fig.add_subplot(235)
    ax5.axis('off')
    rgb_text = "RGB Statistik:\n"
    for channel, stats in rgb_stats.items():
        rgb_text += f"\n{channel}:\n"
        rgb_text += f"  Mean: {stats['mean']:.2f}\n"
        rgb_text += f"  Median: {stats['median']:.2f}\n"
        rgb_text += f"  Std: {stats['std']:.2f}\n"
    ax5.text(0, 0.5, rgb_text, fontsize=10, verticalalignment='center')
    
    # Vis HSV statistik
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    hsv_text = "HSV Statistik:\n"
    for channel, stats in hsv_stats.items():
        hsv_text += f"\n{channel}:\n"
        hsv_text += f"  Mean: {stats['mean']:.2f}\n"
        hsv_text += f"  Median: {stats['median']:.2f}\n"
        hsv_text += f"  Std: {stats['std']:.2f}\n"
    ax6.text(0, 0.5, hsv_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def analyze_terrain_types(tile_labels_mapping, tiles_dir):
    """
    Analyserer farveegenskaber for forskellige terræntyper.
    
    Args:
        tile_labels_mapping: Dictionary med tile-labels
        tiles_dir: Sti til mappen med tile-billeder
    
    Returns:
        dict: Dictionary med farvefeatures grupperet efter terræntype
    """
    # Gruppér tiles efter terræntype
    terrain_tiles = defaultdict(list)
    
    # Gennemgå alle brætter
    for board_name, tiles in tile_labels_mapping.items():
        # Gennemgå alle tiles på dette bræt
        for tile_pos, tile_info in tiles.items():
            terrain = tile_info["terrain"]
            filename = tile_info["filename"]
            
            # Tilføj til terrain_tiles dictionary
            terrain_tiles[terrain].append(filename)
    
    # Beregn farvefeatures for hver terræntype
    terrain_features = {}
    
    for terrain, filenames in terrain_tiles.items():
        print(f"Analyserer {len(filenames)} tiles for terræntype: {terrain}")
        
        # Spring over Home og Unknown
        if terrain in ['Home', 'Unknown']:
            continue
        
        # Indsaml farvestatistik for alle tiles af denne terræntype
        rgb_stats_list = []
        hsv_stats_list = []
        rgb_histograms = []
        hsv_histograms = []
        patch_histograms = []
        
        # Sample nogle billeder til visualisering (max 5)
        sample_images = []
        for i, filename in enumerate(filenames[:5]):
            filepath = os.path.join(tiles_dir, filename)
            # Indlæs billedet
            image = cv2.imread(filepath)
            if image is None:
                print(f"Advarsel: Kunne ikke indlæse {filepath}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sample_images.append(image)
            
        # Beregn statistik for alle tiles
        for filename in filenames:
            filepath = os.path.join(tiles_dir, filename)
            # Indlæs billedet
            image = cv2.imread(filepath)
            if image is None:
                print(f"Advarsel: Kunne ikke indlæse {filepath}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Beregn RGB-statistik
            rgb_stats = compute_rgb_statistics(image)
            rgb_stats_list.append(rgb_stats)
            
            # Beregn HSV-statistik
            hsv_stats = compute_hsv_statistics(image)
            hsv_stats_list.append(hsv_stats)
            
            # Beregn histogrammer
            rgb_hist, _, _ = compute_color_histogram(image, "rgb")
            hsv_hist, _, _ = compute_color_histogram(image, "hsv")
            rgb_histograms.append(rgb_hist)
            hsv_histograms.append(hsv_hist)
            
            # Beregn patch-histogrammer (kan være meget hukommelseskrævende)
            # patches = compute_patch_histograms(image)
            # patch_histograms.append(patches)
        
        # Beregn gennemsnit af statistik
        avg_rgb_stats = {
            'R': {stat: np.mean([tile_stats['R'][stat] for tile_stats in rgb_stats_list]) 
                  for stat in ['mean', 'median', 'std']},
            'G': {stat: np.mean([tile_stats['G'][stat] for tile_stats in rgb_stats_list]) 
                  for stat in ['mean', 'median', 'std']},
            'B': {stat: np.mean([tile_stats['B'][stat] for tile_stats in rgb_stats_list]) 
                  for stat in ['mean', 'median', 'std']}
        }
        
        avg_hsv_stats = {
            'H': {stat: np.mean([tile_stats['H'][stat] for tile_stats in hsv_stats_list]) 
                  for stat in ['mean', 'median', 'std']},
            'S': {stat: np.mean([tile_stats['S'][stat] for tile_stats in hsv_stats_list]) 
                  for stat in ['mean', 'median', 'std']},
            'V': {stat: np.mean([tile_stats['V'][stat] for tile_stats in hsv_stats_list]) 
                  for stat in ['mean', 'median', 'std']}
        }
        
        # Beregn gennemsnitlige histogrammer
        avg_rgb_histogram = np.mean(rgb_histograms, axis=0)
        avg_hsv_histogram = np.mean(hsv_histograms, axis=0)
        
        # Gem resultaterne
        terrain_features[terrain] = {
            'count': len(filenames),
            'rgb_stats': avg_rgb_stats,
            'hsv_stats': avg_hsv_stats,
            'avg_rgb_histogram': avg_rgb_histogram.tolist(),
            'avg_hsv_histogram': avg_hsv_histogram.tolist(),
            'sample_images': sample_images
        }
    
    return terrain_features

def visualize_terrain_features(terrain_features, output_dir='Visualiseringer'):
    """
    Visualiserer farveegenskaber for forskellige terræntyper og gemmer figurerne med meningsfulde navne.
    
    Args:
        terrain_features: Dictionary med farvefeatures grupperet efter terræntype
        output_dir: Mappe hvor visualiseringerne skal gemmes
    """
    # Opret output-mappe hvis den ikke findes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Oprettet mappe: {output_dir}")
    
    terrains = list(terrain_features.keys())
    
    # 1. Visualiser RGB-statistik
    rgb_means = {
        'R': [terrain_features[t]['rgb_stats']['R']['mean'] for t in terrains],
        'G': [terrain_features[t]['rgb_stats']['G']['mean'] for t in terrains],
        'B': [terrain_features[t]['rgb_stats']['B']['mean'] for t in terrains]
    }
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(terrains))
    width = 0.25
    
    plt.bar(x - width, rgb_means['R'], width, label='R', color='red', alpha=0.7)
    plt.bar(x, rgb_means['G'], width, label='G', color='green', alpha=0.7)
    plt.bar(x + width, rgb_means['B'], width, label='B', color='blue', alpha=0.7)
    
    plt.title('Gennemsnitlig RGB-værdi for hver terræntype')
    plt.xticks(x, terrains, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Gem figuren med et beskrivende navn
    rgb_means_filename = os.path.join(output_dir, 'rgb_means_comparison.png')
    plt.savefig(rgb_means_filename, dpi=300)
    print(f"Gemt RGB-sammenligning til: {rgb_means_filename}")
    plt.close()
    
    # 2. Visualiser HSV-statistik
    hsv_means = {
        'H': [terrain_features[t]['hsv_stats']['H']['mean'] for t in terrains],
        'S': [terrain_features[t]['hsv_stats']['S']['mean'] for t in terrains],
        'V': [terrain_features[t]['hsv_stats']['V']['mean'] for t in terrains]
    }
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, hsv_means['H'], width, label='H', color='red', alpha=0.7)
    plt.bar(x, hsv_means['S'], width, label='S', color='green', alpha=0.7)
    plt.bar(x + width, hsv_means['V'], width, label='V', color='blue', alpha=0.7)
    
    plt.title('Gennemsnitlig HSV-værdi for hver terræntype')
    plt.xticks(x, terrains, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Gem figuren med et beskrivende navn
    hsv_means_filename = os.path.join(output_dir, 'hsv_means_comparison.png')
    plt.savefig(hsv_means_filename, dpi=300)
    print(f"Gemt HSV-sammenligning til: {hsv_means_filename}")
    plt.close()
    
    # 3. Visualiser eksempel-billeder for hver terræntype
    cols = min(5, len(terrains))
    rows = (len(terrains) + cols - 1) // cols
    
    plt.figure(figsize=(15, rows * 3))
    for i, terrain in enumerate(terrains):
        # Vis eksempelbillede
        plt.subplot(rows, cols, i+1)
        sample_images = terrain_features[terrain]['sample_images']
        if sample_images:
            plt.imshow(sample_images[0])
        plt.title(f"{terrain} (n={terrain_features[terrain]['count']})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Eksempler på terræntyper", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Gem figuren med et beskrivende navn
    samples_filename = os.path.join(output_dir, 'terrain_samples_overview.png')
    plt.savefig(samples_filename, dpi=300)
    print(f"Gemt terræneksempler til: {samples_filename}")
    plt.close()
    
    # Gem også individuelle eksempelbilleder for hver terræntype
    samples_dir = os.path.join(output_dir, 'individual_samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    for terrain in terrains:
        sample_images = terrain_features[terrain]['sample_images']
        if sample_images:
            # Gem første eksempel for hver terræntype
            sample_filename = os.path.join(samples_dir, f'{terrain}_sample.png')
            plt.figure(figsize=(5, 5))
            plt.imshow(sample_images[0])
            plt.title(f"{terrain}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(sample_filename, dpi=300)
            plt.close()
            
            # Hvis der er flere eksempelbilleder, gem dem også
            for j, img in enumerate(sample_images[1:], 1):
                extra_sample_filename = os.path.join(samples_dir, f'{terrain}_sample{j+1}.png')
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.title(f"{terrain} (eksempel {j+1})")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(extra_sample_filename, dpi=300)
                plt.close()
    
    print(f"Gemt individuelle terræneksempler til: {samples_dir}")
    
    # 4. Visualiser farvehistogrammer for hver terræntype
    histograms_dir = os.path.join(output_dir, 'histograms')
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)
    
    for terrain in terrains:
        plt.figure(figsize=(15, 5))
        
        # RGB histogram
        plt.subplot(121)
        rgb_hist = np.array(terrain_features[terrain]['avg_rgb_histogram'])
        bins = np.arange(len(rgb_hist[0]))
        
        plt.bar(bins, rgb_hist[0], alpha=0.7, color='red', label='R')
        plt.bar(bins, rgb_hist[1], alpha=0.7, color='green', label='G')
        plt.bar(bins, rgb_hist[2], alpha=0.7, color='blue', label='B')
        
        plt.title(f"RGB Histogram for {terrain}")
        plt.legend()
        
        # HSV histogram
        plt.subplot(122)
        hsv_hist = np.array(terrain_features[terrain]['avg_hsv_histogram'])
        bins = np.arange(len(hsv_hist[0]))
        
        plt.bar(bins, hsv_hist[0], alpha=0.7, color='red', label='H')
        plt.bar(bins, hsv_hist[1], alpha=0.7, color='green', label='S')
        plt.bar(bins, hsv_hist[2], alpha=0.7, color='blue', label='V')
        
        plt.title(f"HSV Histogram for {terrain}")
        plt.legend()
        
        plt.tight_layout()
        
        # Gem figuren med et beskrivende navn
        hist_filename = os.path.join(histograms_dir, f'{terrain}_histograms.png')
        plt.savefig(hist_filename, dpi=300)
        plt.close()
    
    print(f"Gemt histogrammer for hver terræntype til: {histograms_dir}")
    
    # Opret en tabel med nøglestatistik for hver terræntype
    stats_table_filename = os.path.join(output_dir, 'terrain_statistics.txt')
    with open(stats_table_filename, 'w') as f:
        f.write("Terræntype statistik - Oversigt\n")
        f.write("=" * 80 + "\n\n")
        
        for terrain in terrains:
            f.write(f"Terræntype: {terrain}\n")
            f.write(f"Antal samples: {terrain_features[terrain]['count']}\n")
            
            f.write("\nRGB Statistik:\n")
            for channel in ['R', 'G', 'B']:
                stats = terrain_features[terrain]['rgb_stats'][channel]
                f.write(f"  {channel}: Middelværdi={stats['mean']:.2f}, Median={stats['median']:.2f}, Std.afv={stats['std']:.2f}\n")
            
            f.write("\nHSV Statistik:\n")
            for channel in ['H', 'S', 'V']:
                stats = terrain_features[terrain]['hsv_stats'][channel]
                f.write(f"  {channel}: Middelværdi={stats['mean']:.2f}, Median={stats['median']:.2f}, Std.afv={stats['std']:.2f}\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"Gemt statistikoversigt til: {stats_table_filename}")

def run_color_analysis(tile_labels_mapping, tiles_dir, output_dir='Visualiseringer'):
    """
    Kører hele farveanalysen for alle terræntyper.
    
    Args:
        tile_labels_mapping: Dictionary med tile-labels
        tiles_dir: Sti til mappen med tile-billeder
        output_dir: Mappe hvor visualiseringerne skal gemmes
    
    Returns:
        dict: Dictionary med farvefeatures grupperet efter terræntype
    """
    print("Starter farveanalyse af terræntyper...")
    terrain_features = analyze_terrain_types(tile_labels_mapping, tiles_dir)
    
    print("Visualiserer resultater...")
    visualize_terrain_features(terrain_features, output_dir)
    
    return terrain_features

# Hvis filen køres direkte
if __name__ == "__main__":
    # Stier
    tiles_dir = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
    tile_labels_file = "tile_labels_mapping.json"
    output_dir = "Visualiseringer"
    
    # Hvis tile_labels_mapping.json findes, indlæs den og kør analysen
    if os.path.exists(tile_labels_file):
        with open(tile_labels_file, 'r') as f:
            tile_labels_mapping = json.load(f)
        
        # Kør farveanalyse
        terrain_features = run_color_analysis(tile_labels_mapping, tiles_dir, output_dir)
        
        # Gem resultaterne til senere brug
        features_file = os.path.join(output_dir, 'terrain_color_features.json')
        with open(features_file, 'w') as f:
            # Konverter sample_images til filnavne for at kunne gemme som JSON
            serializable_features = {}
            for terrain, features in terrain_features.items():
                serializable_features[terrain] = features.copy()
                serializable_features[terrain].pop('sample_images', None)  # Fjern billeder før gem
            
            json.dump(serializable_features, f, indent=2)
        
        print(f"Farveanalyse fuldført. Resultater gemt i {features_file}")
    else:
        print(f"Fil {tile_labels_file} ikke fundet. Kør først labels.py for at oprette denne fil.")
        
        # Vis eksempel på farveanalyse for et enkelt eksempelbillede
        if os.path.exists(tiles_dir) and len(os.listdir(tiles_dir)) > 0:
            example_file = os.path.join(tiles_dir, os.listdir(tiles_dir)[0])
            example_image = cv2.imread(example_file)
            if example_image is not None:
                example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
                
                print(f"Viser farveanalyse for eksempelbillede: {os.path.basename(example_file)}")
                fig = visualize_color_analysis(example_image, f"Farveanalyse: {os.path.basename(example_file)}")
                
                # Gem også dette eksempel
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                example_filename = os.path.join(output_dir, 'single_tile_analysis.png')
                fig.savefig(example_filename, dpi=300)
                print(f"Gemt enkelt-analyse til: {example_filename}")
                
                plt.show()
            else:
                print(f"Kunne ikke indlæse eksempelbilledet: {example_file}")
        else:
            print(f"Ingen billeder fundet i {tiles_dir}")