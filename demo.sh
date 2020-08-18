for defencer in SpatialSmoothing JPEGCompression
do
for epsilon in 0.03137 
do
for attack in PGD
do  
	python main_cifar10_pgd.py --test_ssim --epsilon $epsilon --attack_method $attack  --defence_method $defencer | tee ./log/ssim_$attack\_$epsilon\_$defencer.txt
done
done
done  
