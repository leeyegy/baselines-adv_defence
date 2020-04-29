for defencer in SpatialSmoothing JPEGCompression
do
for epsilon in 0.00784
do
for attack in CW 
do  
	python main_cifar10_pgd.py --epsilon $epsilon --attack_method $attack  --defence_method $defencer | tee ./log/black_box/vgg16/$attack\_$epsilon\_$defencer.txt
done
done
done  
