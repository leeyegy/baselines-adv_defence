for defencer in SpatialSmoothing JPEGCompression
do
for epsilon in 0.00784 0.03137 0.06275
do
for attack in BIM
do  
	python main_cifar10_pgd.py --epsilon $epsilon --attack_method $attack  --defence_method $defencer | tee ./log/$attack\_$epsilon\_$defencer.txt
done
done
done  
