# Detaljeret Gennemgang af Features for King Domino-Projektet

## Farvebaserede Features

### 1. RGB Farvekomponenter
- **Konkret anvendelse**: King Domino har distinkte farveskemaer for hver terræntype (skov: grøn, ager: gul, mine: mørk, vand: blå)
- **Hvad det fortæller os**: Statistiske mål (gennemsnit, median, standardafvigelse) for hver RGB-kanal kan direkte indikere terræntypen
- **Relevans**: ★★★★★ Meget høj - dette er den mest direkte og simple metode til identifikation af terræntyper
- **Fra materialet**: "RGB-farverum: Grundlæggende computerfarverum med 3 primærfarver: Rød, Grøn, Blå" (Materiale/2.txt)

### 2. HSV Farverum
- **Konkret anvendelse**: HSV adskiller kulør (H) fra mætning (S) og lysstyrke (V), hvilket gør det mere robust ved varierende belysning
- **Hvad det fortæller os**: Hue kan indikere terræntype uafhængigt af lysstyrke, hvilket er ideelt hvis billederne har varierende belysning
- **Relevans**: ★★★★☆ Høj - især hvis der er lysforskelle på tværs af brætspilsbilleder
- **Fra materialet**: "Ved at separere farvetone fra lysstyrke bliver det lettere at sætte thresholds, fx for at finde objekter med en bestemt farve under varierende belysning" (Materiale/1.txt)

### 3. Farvehistogrammer
- **Konkret anvendelse**: Forskellige terræntyper vil have karakteristiske farvefordelinger
- **Hvad det fortæller os**: Fordelingen og koncentrationen af farver kan skelne mellem terræntyper med lignende primærfarver men forskellige nuancer
- **Relevans**: ★★★★☆ Høj - kan fange subtile forskelle som simple gennemsnit ikke viser
- **Fra materialet**: "Histogram features omdanner rå data til strukturerede numeriske vektorer" (Materiale/5.txt)

### 4. Dominerende Farver
- **Konkret anvendelse**: K-means clustering kan identificere de 2-3 mest fremtrædende farver i en tile
- **Hvad det fortæller os**: Dominerende farver og deres proportioner giver en kompakt beskrivelse; kroner vil typisk have guldfarve
- **Relevans**: ★★★☆☆ Moderat - særligt nyttig for terræner med flere distinkte farvekomponenter
- **Fra materialet**: "K-means algoritmen grupperer data ved at prøve at adskille samples i n grupper med lige stor varians" (Materiale/8.txt)

## Teksturbaserede Features

### 1. Kantanalyse
- **Konkret anvendelse**: Forskellige terræntyper har forskellige teksturer og kantmønstre
- **Hvad det fortæller os**: Antallet og styrken af kanter kan skelne mellem glatte terræner (vand, enge) og strukturerede (skov, bjerge)
- **Relevans**: ★★★★☆ Høj - særligt til terræntyper med distinkte teksturforskelle
- **Fra materialet**: "En kant er et område i billedet, hvor der sker en markant ændring i intensitet" (Materiale/1.txt)

### 2. Histogram of Oriented Gradients (HOG)
- **Konkret anvendelse**: HOG fanger strukturelle elementer i terræner som træer i skove eller bølger i vand
- **Hvad det fortæller os**: Retning og intensitet af gradienter afslører distinkte mønstre i forskellige terræntyper
- **Relevans**: ★★★★☆ Høj - særligt for terræner med strukturelle mønstre og kroner
- **Fra materialet**: "HOG-algoritmen: 1) Normalisér farver, 2) Beregn gradienter, 3) Vægtet afstemning i rumlige og orienterings-celler" (Materiale/3.txt)

### 3. Local Binary Patterns (LBP)
- **Konkret anvendelse**: LBP kan fange lokale teksturforskelle, særligt ved terrængrænser
- **Hvad det fortæller os**: Lokale teksturmønstre uafhængigt af lysændringer
- **Relevans**: ★★★☆☆ Moderat - kan supplere HOG for visse terræntyper
- **Fra materialet**: Selvom ikke direkte nævnt, bygger det på samme principper om teksturanalyse beskrevet i Materiale/6.txt

### 4. Gradienthistogrammer
- **Konkret anvendelse**: Histogrammer over gradientretninger fanger karakteristiske teksturmønstre
- **Hvad det fortæller os**: Fordelingen af retninger kan indikere om terrænet har horisontale (vand), vertikale (skov) eller kaotiske (bjerge) mønstre
- **Relevans**: ★★★★☆ Høj - særligt som komplement til farveanalyse
- **Fra materialet**: "Tekstur beskrives som et mønster over et billedepatch, typisk opsummeret ved et histogram af gradientorienteringer" (Materiale/6.txt)

## Statistiske Features

### 1. Grundlæggende Statistik
- **Konkret anvendelse**: Statistiske mål for pixelintensiteter i hver tile
- **Hvad det fortæller os**: Middelværdi (generel lyshed), varians (teksturkompleksitet), skævhed/kurtosis (intensitetsfordeling)
- **Relevans**: ★★★☆☆ Moderat - simple indikatorer, der er lette at beregne
- **Fra materialet**: "Histogrammer viser fordelingen af pixelværdier i et billede" (Materiale/1.txt)

### 2. Entropi og Kompleksitetsmål
- **Konkret anvendelse**: Informationsentropi afslører kompleksitet i teksturen
- **Hvad det fortæller os**: Høj entropi indikerer komplekse terræner (skov, bjerge), lav entropi indikerer ensartede (vand, enge)
- **Relevans**: ★★★☆☆ Moderat - nyttigt supplement til andre features
- **Fra materialet**: Relaterer til konceptet om informationsindhold beskrevet indirekte i Materiale/5.txt

### 3. Autocorrelation
- **Konkret anvendelse**: Måler hvor meget et terrænmønster gentager sig selv
- **Hvad det fortæller os**: Høj autocorrelation indikerer regelmæssige mønstre (ager), lav indikerer tilfældige (vand)
- **Relevans**: ★★☆☆☆ Lav til moderat - mere avanceret og potentielt beregningstung
- **Fra materialet**: "Auto-korrelations overfladen hjælper med at bedømme, hvor lokalt stabil et patch er" (Materiale/3.txt)

## Specifikke Features for King Domino

### 1. Kronedetektering
- **Konkret anvendelse**: Direkte identifikation af kroner, afgørende for pointberegning
- **Hvad det fortæller os**: Tilstedeværelsen og antallet af kroner på en tile
- **Relevans**: ★★★★★ Meget høj - central del af opgaven
- **Fra materialet**: "Template matching... en metode til at finde en bestemt form eller et mønster i et billede" (Materiale/1.txt)

### 2. Territoriegrænser og Sammenhæng
- **Konkret anvendelse**: Identifikation af sammenhængende terrænområder
- **Hvad det fortæller os**: Definerer territorier, som er basis for pointberegning
- **Relevans**: ★★★★★ Meget høj - direkte knyttet til spillets pointmekanik
- **Fra materialet**: Connected Component Analysis (Materiale/3.txt) og DBSCAN clustering (Materiale/7.txt)

### 3. Home-feltsdetektering
- **Konkret anvendelse**: Identifikation af slottet som fungerer som joker-element
- **Hvad det fortæller os**: Positionen af slottet, der kan forbinde forskellige terræntyper
- **Relevans**: ★★★★☆ Høj - specifik regel i spillet der påvirker pointberegning
- **Fra materialet**: Kombination af farve- og formbaserede features beskrevet på tværs af materialet

## Segmentering og Klassifikation

### 1. Thresholding
- **Konkret anvendelse**: Adskiller terræntyper baseret på farvetærskler
- **Hvad det fortæller os**: Definerer grænser mellem terræntyper med distinkte farveforskelle
- **Relevans**: ★★★★☆ Høj - grundlæggende metode til segmentering
- **Fra materialet**: "Ved at sætte en grænseværdi (threshold) kan vi klassificere pixels" (Materiale/1.txt)

### 2. K-means Segmentering
- **Konkret anvendelse**: Grupperer lignende pixelværdier til homogene segmenter
- **Hvad det fortæller os**: Identificerer områder med lignende visuelle egenskaber (samme terræntype)
- **Relevans**: ★★★★☆ Høj - effektiv til automatisk segmentering
- **Fra materialet**: "K-means clustering er en kraftfuld algoritme til at gruppere data" (Materiale/8.txt)

### 3. Supervised Learning Pipeline
- **Konkret anvendelse**: Klassifikator til at genkende terræntyper og kroner baseret på features
- **Hvad det fortæller os**: Sandsynlighedsfordeling over terræntyper for hver tile
- **Relevans**: ★★★★☆ Høj - endelig klassifikationsmetode efter feature extraction
- **Fra materialet**: Bygger på principper fra flere af materialerne omkring features og klassifikation

## Dimensionsreduktion

### 1. PCA
- **Konkret anvendelse**: Reducerer dimensionaliteten af feature-vektorer fra tile-analysen
- **Hvad det fortæller os**: Identificerer de mest informative kombinationer af features, reducerer støj
- **Relevans**: ★★★☆☆ Moderat - nyttig hvis vi ender med mange overlappende features
- **Fra materialet**: "PCA er en ikke-superviseret dimensionsreduktionsteknik" (Materiale/4.txt)

### 2. LDA
- **Konkret anvendelse**: Finder feature-dimensioner der bedst adskiller terræntyper
- **Hvad det fortæller os**: Fremhæver aspekter af data der er mest diskriminative mellem terrænklasser
- **Relevans**: ★★★☆☆ Moderat til høj - nyttig hvis vi har labeled træningsdata
- **Fra materialet**: "LDA er en superviseret dimensionsreduktionsteknik, der fokuserer på at maksimere klasseseparation" (Materiale/4.txt)

## Samlet vurdering

For et effektivt King Domino-terræn-analyseprogram anbefaler jeg at fokusere på denne kombination af features:

1. **Højest prioritet**:
   - RGB/HSV farveanalyse (gennemsnit, histogrammer)
   - HOG for teksturidentifikation
   - Template matching for kronedetektering
   - Connected Component Analysis for territoriegrænser

2. **Moderat prioritet**:
   - Kantanalyse for kompleksitetsmåling
   - K-means clustering for segmentering
   - Grundlæggende statistiske mål

3. **Lavere prioritet**:
   - Autocorrelation
   - Avancerede dimensionsreduktionsteknikker
   - Komplekse entropi-målinger

Denne strategi balancerer implementeringskompleksitet med nytten af de forskellige features for det specifikke problem.