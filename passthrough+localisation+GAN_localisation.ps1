# Passthroughs + Passthrough Localisation:
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 12000 | Out-File -FilePath "analysis\lowpass12k.txt"  -Encoding utf8
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 10000 | Out-File -FilePath "analysis\lowpass10k.txt"  -Encoding utf8
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 8000  | Out-File -FilePath "analysis\lowpass8k.txt"   -Encoding utf8
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 6000  | Out-File -FilePath "analysis\lowpass6k.txt"   -Encoding utf8
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 4000  | Out-File -FilePath "analysis\lowpass4k.txt"   -Encoding utf8
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 2000  | Out-File -FilePath "analysis\lowpass2k.txt"   -Encoding utf8
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 1000  | Out-File -FilePath "analysis\lowpass1k.txt"   -Encoding utf8
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 500   | Out-File -FilePath "analysis\lowpass0.5k.txt" -Encoding utf8

# HRTF Selection
python main.py hrtf_selection_baseline --hpc False --type filter --filtertype lowpass --cutfreq 2000 | Out-File -FilePath "analysis\hrtf_selection.txt" -Encoding utf8

# GAN Localisations
python main.py localisation_evaluations --hpc False