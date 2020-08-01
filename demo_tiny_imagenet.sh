for epsilon in 0.0
do
for attack in NONE
do  
for defence in JPEGCompression SpatialSmoothing
do
	python main_tiny_imagenet.py  --epsilon $epsilon --attack_method $attack  --defence_method $defence | tee ./log/tiny_imagenet_$attack\_$epsilon\_$defence.txt
done
done  
done
