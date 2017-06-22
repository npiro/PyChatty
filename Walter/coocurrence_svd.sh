#!/usr/bin/env bash
set -eu # strict config (non-zero returns and unset vars halt script)

#---------------------------------------------
# Script to calculate co-ocurrence statistics
#---------------------------------------------

cd "/home/walter/Dropbox/S2DS - M&S/DevCode/MS/Walter/cca-master"

python cca.py --corpus "/home/walter/Dropbox/S2DS - M&S/Data/Walter/agentMessages.txt" --cutoff 1 --window 10

python cca.py --stat "/home/walter/Dropbox/S2DS - M&S/Data/Walter/agentMessages.cutoff1.window10" --no_matlab --m 2 --kappa 2


python cca.py --corpus "/home/walter/Dropbox/S2DS - M&S/Data/Walter/clientMessages.txt" --cutoff 1 --window 10

python cca.py --stat "/home/walter/Dropbox/S2DS - M&S/Data/Walter/clientMessages.cutoff1.window10" --no_matlab --m 2 --kappa 2
