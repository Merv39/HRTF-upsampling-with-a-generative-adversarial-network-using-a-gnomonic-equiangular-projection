# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 12000 > analysis\lowpass12k.txt
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 10000 > analysis\lowpass10k.txt
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 8000 > analysis\lowpass8k.txt
python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 6000 > analysis\lowpass6k.txt
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 4000 > analysis\lowpass4k.txt
# python main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 2000 > analysis\lowpass2k.txt

# python main.py hrtf_selection_baseline --hpc False --type filter --filtertype lowpass --cutfreq 2000 > analysis\hrtf_selection.txt