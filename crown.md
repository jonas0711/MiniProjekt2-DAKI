# Rangering og analyse af løsninger til kronedetektering

Efter analyse af billederne og en grundig vurdering baseret på undervisningsmaterialet, rangerer jeg løsningerne til kronedetektering som følger:

## 1. Farvebaseret segmentering + Blob Analysis (Højest prioritet)

**Begrundelse:** Kronerne har en karakteristisk gylden farve, som er distinkt selv på det gule terræn i billede 1. Dette matcher præcist med anvendelsen af farvetresholding som beskrevet i Materiale/2.txt og Materiale/4.txt.

**Konkret implementering:**
1. **HSV-transformation**: Konverter billeder til HSV-farverum, hvor det er lettere at isolere kronernes farve
2. **Color thresholding**: Definer et HSV-farveinterval for den gyldne farve
   - H: ~25-45 (gulligt område)
   - S: Høj (>150) for at sikre mættet gylden farve
   - V: Høj (>150) for at fange det lyse guldagtige
3. **Morfologiske operationer**: Anvend erosion/dilation for at fjerne støj
4. **Connected Component Analysis**: Find sammenhængende områder vha. 8-connectivity
5. **BLOB-filtrering**: Filtrer udfra:
   - Areal: Mellem 100-1000 pixels (estimeret fra billederne)
   - Cirkularity: >0.6 (kronerne er nogenlunde cirkulære)
   - Aspect ratio: ~0.8-1.2 (kronerne er næsten lige så høje som brede)

**Fordele:** Simpel, effektiv og direkte baseret på kronernes mest karakteristiske egenskab - deres farve.

## 2. Template Matching + HSV-filtrering

**Begrundelse:** Kronerne har ensartet udseende på tværs af billederne, hvilket gør template matching velegnet. Dette svarer til teknikken beskrevet i Materiale/1.txt.

**Konkret implementering:**
1. **Skabelonoprettelse**: Udskær 3-4 repræsentative kroner fra træningsbilleder
2. **HSV-forfiltrering**: Begræns søgningen til gullige områder (reducerer falske positiver)
3. **Multi-scale matching**: Anvend normalized cross-correlation ved forskellige skaleringer (0.8-1.2x)
4. **Score threshold**: Sæt en tærskelværdi (~0.7) for matchscore

**Fordele:** Kan håndtere varierende belysning, mindre følsom over for farveforskelle når kronerne sidder på forskellige terræntyper.

## 3. Edge Detection + Form Analysis

**Begrundelse:** Kronernes karakteristiske form med zikzak-kanter kan udnyttes, som vist i Materiale/6.txt om kantdetektion.

**Konkret implementering:**
1. **Kantdetektion**: Anvend Canny kantdetektor med parametre (100, 200)
2. **Konturekstraktion**: Find lukkede konturer
3. **Formanalyse**: Beregn:
   - Antal spidser (peaks) - kroner har typisk 5-7 peaks
   - Forhold mellem perimeter og areal
   - Konveks hull-analyse

**Fordele:** Fungerer potentielt ved svær farveadskillelse, fx på gult terræn.

## 4. Histogram of Oriented Gradients (HOG)

**Begrundelse:** HOG kan fange kronernes karakteristiske gradientmønstre, som beskrevet i Materiale/3.txt.

**Konkret implementering:**
1. **Sliding window**: Anvend et vindue på 32x32 pixels
2. **HOG-beregning**: Beregn features for hver position
3. **KNN/SVM-klassifikator**: Træn klassifikator med positive/negative eksempler
4. **Non-Maximum Suppression**: Fjern overlappende detektioner

**Fordele:** Meget robust over for farve- og belysningsvariation, men mere kompleks implementering.

## 5. Kombination (Ensemble) metode

**Begrundelse:** Kombinerer styrkerne fra flere metoder for at øge robustheden.

**Konkret implementering:**
1. **Parallelle detektorer**: Kør både farvebaseret metode og formbaseret metode
2. **Vægtet stemmegivning**: Kombiner resultater med vægtning
3. **Spatial clustering**: Sammenhold detektioner der er tæt på hinanden

**Fordele:** Højest robusthed, men også mest beregningstung.

## Konklusion

Den bedste tilgang vil være **farvebaseret segmentering + blob analysis** grundet:
1. **Effektivitet**: Direkte baseret på kronernes mest iøjnefaldende karakteristik (farven)
2. **Simplicitet**: Kræver minimal træning og parameteroptimering
3. **Dokumenteret effektivitet**: Lignende teknikker i undervisningsmaterialet har vist sig effektive til lignende opgaver
4. **Skalerbarhed**: Kan udvides med formbaseret analyse hvis der opstår udfordringer

I særlige tilfælde hvor farveadskillelse er vanskelig (som på det gule terræn i billede 1), kan dette suppleres med kantdetektion for bedre robusthed.