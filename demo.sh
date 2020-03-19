for epsilon in 0.00784  0.03137 0.06275
do 
	python main_cifar10_pgd.py --epsilon $epsilon --test_samples 10000 | tee ./log/wide_pgd_$epsilon\_10000_FeatureSqueezing.txt
done 
