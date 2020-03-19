for epsilon in 2 8 16 
do 
	python main_cifar10_pgd.py --epsilon $epsilon --test_samples 500 --defence_method PixelDefend | tee ./log/wide_pgd_$epsilon\_500_PixelDefend.txt
done 
