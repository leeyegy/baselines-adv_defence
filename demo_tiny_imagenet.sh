for epsilon in 0.00784 0.03137 0.06275 
do
for attack in DeepFool CW
do  
	python main_tiny_imagenet.py  --epsilon $epsilon --attack_method $attack  --defence_method TotalVarMin | tee ./log/tiny_imagenet_$attack\_$epsilon\_TotalVarMin.txt
done
done  

