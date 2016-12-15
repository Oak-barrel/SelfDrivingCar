



#sudo python main.py --lr 0.01 --keep1 0.9 --keep2 0.6 --architecture draw --glimpse 16 --TimeSteps 4
sudo stdbuf -oL python main.py --lr 0.001 --keep1 0.9 --keep2 0.6 --architecture draw --glimpse 16 --TimeSteps 4 1> logs/1.txt 2> logs/1.err &
sudo stdbuf -oL python main.py --lr 0.0001 --keep1 0.9 --keep2 0.6 --architecture draw --glimpse 16 --TimeSteps 4 1> logs/2.txt 2> logs/2.err 

# Fine-grained :- Grid Search
sudo stdbuf -oL python main.py --lr 0.001 --keep1 0.9 --keep2 0.6 --architecture draw --glimpse 8 --TimeSteps 4 1> logs/1.1.txt 2>logs/1.1.err &
sudo stdbuf -oL python main.py --lr 0.001 --keep1 0.9 --keep2 0.6 --architecture draw --glimpse 8 --TimeSteps 8 1> logs/1.2.txt 2>logs/1.2.err
sudo stdbuf -oL python main.py --lr 0.001 --keep1 0.9 --keep2 0.6 --architecture draw --glimpse 12 --TimeSteps 4 1> logs/1.3.txt 2> logs/1.3.err &
sudo stdbuf -oL python main.py --lr 0.001 --keep1 0.9 --keep2 0.6 --architecture draw --glimpse 12 --TimeSteps 8 1> logs/1.4.txt 2> logs/1.4.err






