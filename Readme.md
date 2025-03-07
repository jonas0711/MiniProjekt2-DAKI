# Miniprojekt 2: Automatisk Scoreberegning for King Domino

## Projektbeskrivelse
I dette miniprojekt skal I udvikle et system, der automatisk kan beregne slutscoren for brætspillet King Domino. Systemet skal analysere et billede af en spillers kingdom (spilleområde) og beregne pointene baseret på spillets regler.

King Domino er et brætspil, hvor spillerne bygger små kongeriger af farvede dominobrikker. Vinderen er den, som har flest point når spillet er slut, og pointene afhænger af, hvordan spillerens farvede dominobrikker ligger i forhold til hinanden.

## Spilleregler
King Domino har specifikke regler for pointberegning:

1. Et territorium består af sammenhængende felter af samme terræntype (forbundet vandret eller lodret).
2. For hvert territorium ganges antallet af felter med antallet af kroner i territoriet.
3. Territorier uden kroner giver 0 point, uanset størrelse.
4. Home-feltet (slottet) fungerer som joker, der kan forbinde forskellige terræntyper.
5. Harmony-bonus: 5 ekstra point for et komplet 5x5 grid uden huller.

For detaljerede regler, se de officielle spilleregler eller se videoen om pointgivning.

## Projektformat
Miniprojektet laves i tomandsgrupper. Tast grupperne ind [her](https://link-til-gruppetilmelding) efter første kursusgang.

### Krav til projektet
1. Systemet skal kunne identificere forskellige terræntyper på billedet.
2. Systemet skal korrekt beregne score baseret på spillereglerne.
3. Systemet skal kunne håndtere forskellige layout og konfigurationer.
4. Koden skal være veldokumenteret og struktureret.

### Datasæt
Vi forsyner jer med billeder af spilleplader. Datasættet indeholder:
- Udklippede og perspektivkorrigerede billeder af spilleplader
- Komplette billeder af spillesituationer (til udvidelse af projektet)

### Afleveringskrav
Afleveringen består af:
1. **Kildekode**: Velkommenteret og struktureret kode, der implementerer systemet
2. **Rapport**: En kort rapport (med billeder) der forklarer:
   - Hvordan jeres system virker
   - Hvilke metoder I har anvendt
   - Den matematiske baggrund for jeres løsning
   - Evaluering af systemets præcision

## Tidslinje
- **Afsat tid**: 2 kursusgange (forvent dog at bruge væsentlig længere tid)
- **Deadline**: 28. april 2025 kl. 23:59

## Hints og tips
1. **Start småt**: Brug de udklippede og perspektivkorrigerede spilleplader.
2. **Opbyg en ground truth**: Beregn manuelt hvad scoren er for alle jeres plader.
3. **Opdel jeres data**: Del jeres data i træningssæt (~80%) og testsæt (~20%). Brug træningssættet under udvikling og testsættet til endelig evaluering.
4. **Udvidelsesmulighed**: Hvis det går rigtig godt, kan I udvide systemet, så det selv kan klippe spillepladerne ud af de fulde billeder.


## Bedømmelseskriterier
Projektet bedømmes på:
1. Korrekt implementering af pointberegning
2. Nøjagtighed i identifikation af terræntyper
3. Robusthed over for forskellige layouts
4. Kodekvalitet og dokumentation
5. Rapportens klarhed og grundighed

God arbejdslyst med projektet!
