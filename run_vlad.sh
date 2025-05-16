vlad_py="python3 -u main_vlad.py"

tmux new -d "export export PATH=$HOME/.local/bin:$PATH && export OMP_NUM_THREADS=1 && $vlad_py &> vlad.log"
