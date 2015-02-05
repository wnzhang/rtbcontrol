advs="1458" # "2259 2261 2821 2997 3358 3386 3427 3476"
for adv in $advs; do
    python ../python/control-ecpc-multiex-pid-bid-optimisation.py $adv
done
