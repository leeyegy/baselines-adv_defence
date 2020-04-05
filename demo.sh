for epsilon in 0.00784 0.03137 0.06275 
do
for attack in PGD
do  
	python main_cifar10_pgd.py --epsilon $epsilon --attack_method $attack  --defence_method TotalVarMin | tee ./log/wide_$attack\_$epsilon\_TotalVarMin.txt
done
done  
 python main_cifar10_pgd.py --epsilon 0.0  --attack_method CW  --defence_method TotalVarMin | tee ./log/wide_CW_0.0_TotalVarMin.txt
