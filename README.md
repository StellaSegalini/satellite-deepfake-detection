## Descrizione del Progetto

Le recenti evoluzioni dei modelli generativi consentono la creazione di immagini satellitari sintetiche ad altissima fedeltà visiva. Questo progetto propone un classificatore binario basato su **ResNet50** per distinguere immagini reali acquisite da **fMoW (Functional Map of the World) e SpaceNet** da immagini sintetiche generate tramite **DiffusionSat**.

### Key Features
* **Architettura:** 
Modello Base (ResNet50): Utilizza una rete neurale pre-addestrata, già capace di riconoscere forme, contorni e caratteristiche visive fondamentali.

Fine-Tuning: "Specializza" il modello, riaddestrando i livelli finali sui dati per adattarlo al dominio delle immagini satellitari.

Data Augmentation (Varietà dei dati): Durante l'addestsramento, le immagini vengono modificate in modo dinamico (rota
zioni, ribaltamenti, variazioni di luminosità). Questo evita che il modello impari i dati a memoria (*overfitting*) e lo rende più robusto rispetto a foto storte, scure o con diverse angolazioni.

Analisi di Spiegabilità e Robustezza: Ispezione delle aree decisionali tramite mappe Grad-CAM e valutazione del degrado dell'accuratezza sotto compressione JPEG progressiva.

## Struttura del Progetto

tesi_satellitare/
├── checkpoints/                        # Pesi dei modelli e checkpoint addestrati
│   ├── .cache/                         # Cache locale dei modelli
│   ├── finetune_sd21_.../              # Checkpoint pre-addestrato di DiffusionSat
│   ├── fmow_100/                       # ResNet-50 addestrata su 100% fMoW reale
│   ├── mixed_25_75/                    # ResNet-50 addestrata su 25% fMoW e 75% SpaceNet
│   ├── mixed_50_50/                    # ResNet-50 addestrata su 50% fMoW e 50% SpaceNet
│   ├── mixed_75_25/                    # ResNet-50 addestrata su 75% fMoW e 25% SpaceNet
│   └── spacenet_100/                   # ResNet-50 addestrata su 100% SpaceNet reale
├── data/                               # Dataset per addestramento e test
│   ├── fake/                           # Immagini sintetiche generate per training/val
│   ├── fake_test2/                     # Immagini sintetiche aggiuntive per test
│   ├── real/                           # Immagini reali da fMoW
│   └── spacenet_real/                  # Immagini reali da SpaceNet
├── reports/                            # Output delle valutazioni (matrici, Grad-CAM, grafici JPEG)
│   └── predictions/                    # Output visivi generati dall'inferenza su singola immagine
├── src/                                # Script sorgente di elaborazione ed esecuzione
│   ├── convert_to_png.py               # Conversione e uniformazione delle immagini in formato PNG
│   ├── convert_to_safetensors.py       # Conversione pesi in formato SafeTensors
│   ├── download_real_dataset.py        # Download ed estrazione dei dati reali fMoW
│   ├── download_spacenet.py            # Download ed estrazione dei dati SpaceNet
│   ├── evaluate_all_experiments.py     # Inferenza, Grad-CAM e robustezza JPEG su validation set
│   ├── generate_dataset.py             # Generazione immagini sintetiche con DiffusionSat
│   ├── generate_dataset2.py            # Pipeline alternativa/estesa di generazione dati
│   ├── generate_test.py                # Script rapido per test di generazione singola
│   ├── predict_single_image.py         # Inferenza e classificazione binaria su singola immagine
│   ├── test_unseen_heldout.py          # Valutazione finale su dati non visti (held-out test set)
│   └── train_all_experiments.py        # Pipeline di addestramento ResNet-50 per i 5 scenari
├── README.md                           # Documentazione del repository
└── requirements.txt                    # Dipendenze e librerie Python richieste

# Installazione: 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Utilizzo:

1. Preparazione del Dataset
    Scarica le immagini reali 
python3 src/download_real_dataset.py
python3 src/download_spacenet.py
python3 src/convert_to_png.py

 Genera le immagini fake con DiffusionSat
python3 src/generate_dataset.py
python3 src/generate_dataset2.py

 Per testare una singola generazione rapida:
python3 src/generate_test.py

2. Addestramento modello:
python3 src/train_all_experiments.py

3. Valutazione, Spiegabilità (Grad-CAM) e Stress Test
python3 src/evaluate_all_experiments.py

# Test finale su campioni non utilizzati
python3 src/test_unseen_heldout.py

4. Classificazione singola immagine
python3 src/predict_single_image.py percorso_immagine
