# AryrhytmiaLSTM

Wykorzystanie modeli opartych o warstwy LSTM do klasyfikacji arytmi na podstawie datasetu MIT-BIH, porównanie metod augmentacji danych dla niezbalansowanych klas

## Instalacja i konfiguracja środowiska

### Wymagania

- Python 3.11
- venv

### Tworzenie środowiska

```bash
python3 -m venv .AryrhytmiaEnv

# Windows:
.AryrhytmiaEnv\Scripts\activate
# Linux / macOS:
source .AryrhytmiaEnv/bin/activate
```

### Instalacja zależności

```bash
pip install -r requirements.txt
```

## Dataset

Projekt wykorzystuje bazę [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
dostępną na platformie PhysioNet.

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)

Dataset nie jest dołączony do repozytorium — zostanie automatycznie pobrany
przy pierwszym uruchomieniu notebooka za pomocą biblioteki `wfdb`.

**Licencja datasetu:** [ODC-BY 1.0](https://physionet.org/content/mitdb/1.0.0/)
