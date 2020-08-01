for defencer in SpatialSmoothing JPEGCompression
do
for epsilon in 0.03137 0.06275
do
for iter in 5 10
do  
	python test_BPDA.py --epsilon $epsilon   --defence_method $defencer --max_iterations $iter | tee ./log/BPDA-$iter\_$epsilon\_$defencer.txt
done
done
done  
