# Cetacean - Marine Mammals

## TODO-Tasks

- [ ] SOTA de BirdNET aplicado a cetáceos, el artículo que yo conozco es: Global birdsong embeddings enable superior transfer learning for bioacoustic classification | Scientific Reports (nature.com). @ALL
- [x] Script Segmentar Audios.
    - [x] Pasar ejemplo anotaciones y audios de varios audios diferentes. @Daniel
    - [x] Adaptar Script a Python y compartir. @Alba
- [ ] Segmentar Audios y meter en subcarpetas. Leer punto 8 de kahst/BirdNET-Analyzer: BirdNET analyzer for scientific audio data processing. (github.com) para entender mejor cómo serán procesados los segmentos y cómo deben estar organizadas las carpetas. @Daniel @Neus 
- [ ] Crear CSVs / Excels de los segmentos que forman Train / Val / Test. En proporción 80 / 10 / 10 para cada clase. Los datos tienen que ser independientes y no estar correlacionados. Hacer esta proporción por clase. Si no se consigue un 80-10-10 perfecto no pasa nada si se mantiene esa independencia.
- [x] Reunión. Intento de reunión 14-15 Octubre, por confirmar @Alba.
- [ ] Reunión Prueba de metodología. 21-22 Octubre tener una prueba de resultados con BirdNET. @Alba

## Proposed Pipeline

- **BirdNET Base**: Probar BirdNET para nuestros datos sin entrenar en cetáceos, BirdNET Base, tal y como está ahora para el público – se esperan resultados malos malísimos
- **BirdNET Cetaceans**: Crear nuestro propio BirdNET para cetáceos con el dataset Watkins Marine Mammal Sounds Database (WMMSD) y probar luego como lo hace con nuestros datos – debería mejorar
- **BirdNET SEANIMALMOVE**: Entrenar BirdNET base con nuestros datos y probar – debería mejorar
- **BirdNET Cetaceans + SEANIMALMOVE** Entrenar nuestro BirdNET para cetáceos del dataset WMMSD con nuestros propios datos – debería ser lo que mejor funcione

## Species of Interest

- Cachalote / Sperm Whale (*Physeter microcephalus*)
- Rorcual común / Fin Whale (*Balaenoptera physalus*)
- Calderón común / Pilot Whale (*Globicephala melas*)
- Delfín Común / Common Dolphin (*Delphinus delphis*)
- Delfín listado / Striped Dolphin (*Stenella coeruleoalba*)
- Delfín mular / Common bottlenose dolphin (*Tursiops truncatus*)
- Orca / Killer Whale (*Orcinus orca*)
- Rorcual aliblanco / Minke Whale (*Balaenoptera acutorostrata*)
- Yubarta ó Ballena jorobada / Humpback Whale (*Megaptera novaeangliae*)
