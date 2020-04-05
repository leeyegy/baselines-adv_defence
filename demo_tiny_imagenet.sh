for epsilon in 0.00784 0.03137 0.06275 
do
for attack in PGD FGSM Momentum 
do  
	python main_tiny_imagenet.py  --epsilon $epsilon --attack_method $attack  --defence_method TotalVarMin | tee ./log/tiny_imagenet_$attack\_$epsilon\_TotalVarMin.txt
done
done  

for epsilon in 0.0
do
for attack in STA DeepFool CW none
do  
	python main_tiny_imagenet.py  --epsilon $epsilon --attack_method $attack  --defence_method TotalVarMin | tee ./log/tiny_imagenet_$attack\_$epsilon\_TotalVarMin.txt
done
done  
