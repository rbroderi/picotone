# Listening set metrics (exactly the files in out/listen_lstm/)

`vs stock` = SNR of the file against the stock decode of the same clip.  `d` columns = change against stock, measured
against the source (+ log-mel / + MR-STFT / - SNR are worse).

| set | clip | variant | SNR vs stock dB | SNR vs source | log-mel | MR-STFT | d SNR | d log-mel | d MR-STFT |
|---|---|---|---|---|---|---|---|---|---|
| 48k 24kbps | orchestral | stock |  | 13.65 | 3.98 | 1.42 |  |  |  |
| 48k 24kbps | orchestral | dec448W | 23.4 | 13.50 | 4.06 | 1.43 | -0.15 | +0.08 | +0.01 |
| 48k 24kbps | orchestral | dec384LS | 21.8 | 13.37 | 4.02 | 1.43 | -0.28 | +0.04 | +0.01 |
| 48k 12kbps | orchestral | stock |  | 11.16 | 4.30 | 1.48 |  |  |  |
| 48k 12kbps | orchestral | dec384LS | 20.4 | 10.92 | 4.35 | 1.49 | -0.24 | +0.05 | +0.01 |
| 48k 24kbps | orchestral | dec320LS | 17.9 | 12.97 | 4.12 | 1.45 | -0.68 | +0.14 | +0.03 |
| 48k 24kbps | orchestral | both448W | 14.1 | 13.98 | 4.02 | 1.42 | +0.33 | +0.04 | +0.00 |
| 48k 24kbps | orchestral | both384LS | 13.7 | 14.06 | 3.91 | 1.41 | +0.41 | -0.07 | -0.01 |
| 48k 12kbps | orchestral | both384LS | 11.2 | 11.47 | 4.25 | 1.48 | +0.31 | -0.06 | -0.00 |
| 48k 24kbps | orchestral | both320LS | 12.8 | 14.05 | 3.97 | 1.43 | +0.40 | -0.01 | +0.01 |
| 48k 24kbps | ambient-electronic | stock |  | 12.17 | 3.17 | 1.22 |  |  |  |
| 48k 24kbps | ambient-electronic | dec448W | 25.3 | 12.09 | 3.27 | 1.22 | -0.08 | +0.10 | -0.00 |
| 48k 24kbps | ambient-electronic | dec384LS | 23.2 | 11.94 | 3.34 | 1.23 | -0.23 | +0.17 | +0.01 |
| 48k 12kbps | ambient-electronic | stock |  | 11.80 | 3.28 | 1.24 |  |  |  |
| 48k 12kbps | ambient-electronic | dec384LS | 22.2 | 11.43 | 3.48 | 1.25 | -0.36 | +0.20 | +0.01 |
| 48k 24kbps | ambient-electronic | dec320LS | 18.1 | 11.19 | 3.51 | 1.25 | -0.98 | +0.34 | +0.02 |
| 48k 24kbps | ambient-electronic | both448W | 17.8 | 12.24 | 3.26 | 1.22 | +0.07 | +0.09 | -0.00 |
| 48k 24kbps | ambient-electronic | both384LS | 17.0 | 12.06 | 3.34 | 1.23 | -0.11 | +0.18 | +0.00 |
| 48k 12kbps | ambient-electronic | both384LS | 16.2 | 11.46 | 3.48 | 1.25 | -0.34 | +0.21 | +0.00 |
| 48k 24kbps | ambient-electronic | both320LS | 14.6 | 11.30 | 3.49 | 1.25 | -0.87 | +0.32 | +0.02 |
| 48k 24kbps | speech | stock |  | 8.11 | 4.13 | 1.69 |  |  |  |
| 48k 24kbps | speech | dec448W | 24.4 | 8.02 | 4.08 | 1.69 | -0.09 | -0.04 | +0.00 |
| 48k 24kbps | speech | dec384LS | 21.5 | 7.94 | 4.12 | 1.69 | -0.17 | -0.01 | -0.00 |
| 48k 24kbps | speech | dec320LS | 17.8 | 7.78 | 4.24 | 1.69 | -0.33 | +0.12 | +0.01 |
| 48k 24kbps | speech | both448W | 11.9 | 7.96 | 4.09 | 1.69 | -0.15 | -0.03 | +0.00 |
| 48k 24kbps | speech | both384LS | 11.5 | 7.91 | 4.17 | 1.70 | -0.20 | +0.04 | +0.01 |
| 48k 24kbps | speech | both320LS | 10.7 | 7.78 | 4.34 | 1.70 | -0.33 | +0.22 | +0.01 |
| 48k 24kbps | piano | stock |  | 13.67 | 2.79 | 1.33 |  |  |  |
| 48k 24kbps | piano | dec448W | 24.4 | 13.55 | 2.81 | 1.34 | -0.12 | +0.02 | +0.01 |
| 48k 24kbps | piano | dec384LS | 22.5 | 13.38 | 2.83 | 1.33 | -0.29 | +0.04 | -0.00 |
| 48k 24kbps | piano | dec320LS | 19.0 | 12.85 | 2.89 | 1.35 | -0.82 | +0.10 | +0.01 |
| 48k 24kbps | piano | both448W | 16.4 | 13.54 | 2.83 | 1.33 | -0.13 | +0.04 | -0.00 |
| 48k 24kbps | piano | both384LS | 15.8 | 13.39 | 2.83 | 1.34 | -0.29 | +0.04 | +0.00 |
| 48k 24kbps | piano | both320LS | 14.2 | 12.78 | 2.90 | 1.36 | -0.89 | +0.11 | +0.02 |
| 48k 24kbps | chiptune | stock |  | 9.88 | 6.48 | 2.29 |  |  |  |
| 48k 24kbps | chiptune | dec448W | 22.3 | 9.89 | 6.58 | 2.30 | +0.01 | +0.10 | +0.01 |
| 48k 24kbps | chiptune | dec384LS | 21.3 | 9.82 | 6.68 | 2.31 | -0.06 | +0.20 | +0.02 |
| 48k 24kbps | chiptune | dec320LS | 18.3 | 9.70 | 6.79 | 2.32 | -0.17 | +0.31 | +0.03 |
| 48k 24kbps | chiptune | both448W | 14.5 | 9.90 | 6.53 | 2.29 | +0.02 | +0.05 | +0.01 |
| 48k 24kbps | chiptune | both384LS | 14.2 | 9.79 | 6.66 | 2.31 | -0.09 | +0.18 | +0.02 |
| 48k 24kbps | chiptune | both320LS | 13.3 | 9.71 | 6.78 | 2.31 | -0.17 | +0.30 | +0.02 |
| 24k 6kbps | speech | stock |  | 3.00 | 4.46 | 1.77 |  |  |  |
| 24k 6kbps | speech | dec384LS | 14.5 | 3.01 | 4.61 | 1.79 | +0.01 | +0.16 | +0.02 |
| 24k 6kbps | speech | both384LS | 7.3 | 2.83 | 4.65 | 1.79 | -0.17 | +0.19 | +0.02 |
| 24k 6kbps | orchestral | stock |  | 9.96 | 4.33 | 1.08 |  |  |  |
| 24k 6kbps | orchestral | dec384LS | 19.0 | 9.37 | 4.80 | 1.16 | -0.59 | +0.47 | +0.07 |
| 24k 6kbps | orchestral | both384LS | 11.3 | 9.44 | 4.78 | 1.15 | -0.51 | +0.46 | +0.07 |

## Mean over clips

| set | variant | clips | SNR vs stock dB | d SNR | d log-mel | d MR-STFT | worst clip d log-mel |
|---|---|---|---|---|---|---|---|
| 48k 24kbps | dec448W | 5 | 24.0 | -0.08 | +0.05 | +0.01 | +0.10 |
| 48k 24kbps | dec384LS | 5 | 22.1 | -0.21 | +0.09 | +0.01 | +0.20 |
| 48k 12kbps | dec384LS | 2 | 21.3 | -0.30 | +0.13 | +0.01 | +0.20 |
| 48k 24kbps | dec320LS | 5 | 18.2 | -0.60 | +0.20 | +0.02 | +0.34 |
| 48k 24kbps | both448W | 5 | 14.9 | +0.03 | +0.04 | +0.00 | +0.09 |
| 48k 24kbps | both384LS | 5 | 14.4 | -0.05 | +0.07 | +0.01 | +0.18 |
| 48k 12kbps | both384LS | 2 | 13.7 | -0.01 | +0.07 | +0.00 | +0.21 |
| 48k 24kbps | both320LS | 5 | 13.1 | -0.37 | +0.19 | +0.02 | +0.32 |
| 24k 6kbps | dec384LS | 2 | 16.7 | -0.29 | +0.31 | +0.04 | +0.47 |
| 24k 6kbps | both384LS | 2 | 9.3 | -0.34 | +0.32 | +0.04 | +0.46 |
