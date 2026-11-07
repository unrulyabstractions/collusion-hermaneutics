#!/bin/zsh
# harvest.sh <apple repo dir> <study> <run>...: finished runs from apple to the Mac submodule (sizes compared),
# then their logs and runs to wild's extraction repo. Exit 1 on any mismatch.
set -u
SRC=$1; STUDY=$2; shift 2
SUB=/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
cd $SUB
L=(); for r in "$@"; do L+=(runs/$STUDY/$r logs/$STUDY/$r logs/$STUDY/screen/$r.log); done
mkdir -p runs/$STUDY logs/$STUDY/screen
ssh -o LogLevel=ERROR arena9-apple "cd $SRC && test -f runs/$STUDY/$1/run.json && tar cf - ${L[*]}" | tar xf - || { echo "FAIL apple->mac"; exit 1; }
ssh -o LogLevel=ERROR arena9-apple "cd $SRC && find ${L[*]} -type f -printf '%s %p\n' | sort -k2" > $SCR/h_apple.txt
find ${L[@]} -type f -exec stat -f '%z %N' {} + | sort -k2 > $SCR/h_mac.txt
diff $SCR/h_apple.txt $SCR/h_mac.txt > /dev/null || { echo "FAIL sizes differ"; diff $SCR/h_apple.txt $SCR/h_mac.txt | head; exit 1; }
echo "mac: $(wc -l < $SCR/h_mac.txt) files, sizes match apple"
tar cf - ${L[@]} | ssh -o LogLevel=ERROR arena9-wild "cd /root/work_dontpressure/repo/data/hf/spar-guardian-agents && tar xf - && find ${L[*]} -type f -printf '%s %p\n' | sort -k2" > $SCR/h_wild.txt
diff $SCR/h_mac.txt $SCR/h_wild.txt > /dev/null || { echo "FAIL wild sizes differ"; exit 1; }
echo "wild: $(wc -l < $SCR/h_wild.txt) files, sizes match"
