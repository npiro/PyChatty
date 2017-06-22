#!/usr/bin/env bash
set -eu # strict config (non-zero returns and unset vars halt script)

### set paths
data_dir="'/home/walter/Dropbox/S2DS - M&S/Data'"
input_file_name='04_doc2vecTrainingDataFiltered_noSpellcheckForWalter_first10lines'
#input_file_name='04_doc2vecTrainingDataFiltered'
cmd="python correct_typos.py ${data_dir} $input_file_name"

#echo $cmd
eval "$cmd"
