echo "=== DATE"; date; date +%s
echo "=== HOST"; hostname; uptime
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
df -h / /workspace 2>/dev/null
echo "=== TMUX"; tmux ls 2>&1
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command}' 2>/dev/null | while read t c; do echo "--- PANE $t ($c)"; tmux capture-pane -p -t "$t" -S -40 2>&1 | tail -40; done
echo "=== SCREEN"; screen -ls 2>&1
echo "=== PS"; ps -eo pid,etime,pcpu,pmem,args --sort=-pcpu | head -25
