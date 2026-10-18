#!/bin/zsh
# Resume the Sefaria curated-ref text fetch (skips refs already on disk), then rebuild the derived pairs.
# Network only; no model loading, so it runs outside the guard.
cd /Users/isaac/Documents/GitHub/genizah_search/evals/embedding_audit
A=/Volumes/home/studio_offload/genizah_search_embedding_audit
LOG=$A/logs/sefaria_chain.log
echo "=== $(date) resuming curated ref texts" >> $LOG
python3 sefaria_pull.py texts --refs-file $A/sefaria/derived/curated_refs.txt --rate 3 >> $LOG 2>&1
echo "=== $(date) texts done (exit $?); rebuilding pairs" >> $LOG
python3 sefaria_pairs.py >> $LOG 2>&1
echo "=== $(date) sefaria chain finished (exit $?)" >> $LOG
