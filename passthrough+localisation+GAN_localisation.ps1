# GAN Localisations
# python main.py localisation_evaluations --hpc False

# Passthroughs + Passthrough Localisation:
# python main.py passthrough_baseline --hpc False --type none  | Out-File -FilePath "analysis\None.txt" -Encoding utf8

# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 12000 > analysis\lowpass12k.txt 
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 10000 > analysis\lowpass10k.txt 
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 8000  > analysis\lowpass8k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 6000  > analysis\lowpass6k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 4000  > analysis\lowpass4k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 2000  > analysis\lowpass2k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 1000  > analysis\lowpass1k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 500   > analysis\lowpass0.5k.txt

# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 12000 > analysis\noisylowpass12k.txt 
# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 10000 > analysis\noisylowpass10k.txt 
# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 8000  > analysis\noisylowpass8k.txt  
# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 6000  > analysis\noisylowpass6k.txt  
# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 4000  > analysis\noisylowpass4k.txt  
# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 2000  > analysis\noisylowpass2k.txt  
# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 1000  > analysis\noisylowpass1k.txt  
# python main.py passthrough_baseline --hpc False --type noisyfilter --filtertype lowpass --cutfreq 500   > analysis\noisylowpass0.5k.txt

# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 12000 > analysis\highpass12k.txt 
# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 10000 > analysis\highpass10k.txt 
# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 8000  > analysis\highpass8k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 6000  > analysis\highpass6k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 4000  > analysis\highpass4k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 2000  > analysis\highpass2k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 1000  > analysis\highpass1k.txt  
# python main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 500   > analysis\highpass0.5k.txt

# # HRTF Selection
# python main.py hrtf_selection_baseline --hpc False --type filter --filtertype lowpass --cutfreq 2000 | Out-File -FilePath "analysis\hrtf_selection.txt" -Encoding utf8
