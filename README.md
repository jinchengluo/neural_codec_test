# TODO

- analyse spectrale d'une piste audio et reconstruction spectrale à partir de wavtokenizer
- trouver un dataset pas trop gros et construire un codebook à la main à partir de celui là en initialisant comme dans wavtokenizer
- projeter le dataset dans ce codebook et tokeniser les pistes pour identifier les vecteurs
- projeter le dataset dans le codebook de wavtokenizer et identifier les vecteurs à partir des tokens générés
- inspecter la corrélation entre les deux codebook
- check les perfs avec torchaudio squim et les métriques