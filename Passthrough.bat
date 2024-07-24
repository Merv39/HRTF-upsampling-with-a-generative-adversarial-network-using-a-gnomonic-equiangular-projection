main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 10000 > analysis\lowpass10k.txt
main.py passthrough_baseline --hpc False --type filter --filtertype lowpass --cutfreq 6000 > analysis\lowpass6k.txt

main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 2000 > analysis\highpass2k.txt
main.py passthrough_baseline --hpc False --type filter --filtertype highpass --cutfreq 1000 > analysis\highpass1k.txt

main.py passthrough_baseline --hpc False --type filter --filtertype lowshelf --cutfreq 5000 --gain 12 > analysis\lowshelf5k_12db.txt
main.py passthrough_baseline --hpc False --type filter --filtertype lowshelf --cutfreq 5000 --gain -12 > analysis\lowshelf5k_-12db.txt

main.py passthrough_baseline --hpc False --type filter --filtertype highshelf --cutfreq 5000 --gain 12 > analysis\highshelf5k_12db.txt
main.py passthrough_baseline --hpc False --type filter --filtertype highshelf --cutfreq 5000 --gain -12 > analysis\highshelf5k_-12db.txt

main.py passthrough_baseline --hpc False --type filter --filtertype bandpass --cutfreq 1000 --cutfreq2 10000 > analysis\bandpass1to10k.txt
main.py passthrough_baseline --hpc False --type filter --filtertype bandpass --cutfreq 2000 --cutfreq2 6000 > analysis\bandpass2to6k.txt

main.py passthrough_baseline --hpc False --type filter --filtertype bandstop --cutfreq 5000 --cutfreq2 10000 > analysis\bandstop5to10k.txt
main.py passthrough_baseline --hpc False --type filter --filtertype bandstop --cutfreq 10000 --cutfreq2 15000 > analysis\bandstop10to15k.txt