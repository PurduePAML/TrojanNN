layer=fc8
xy=0
seed=1
unit1=81
unit2=694
neuron=1
filter_shape=0

name="${layer}_${seed}_${unit1}_${unit2}_${neuron}_${filter_shape}"
echo ${unit1} ${name} ${layer} ${xy} ${seed} ${neuron} ${filte_shape} ${unit2}
python ./act_max.tvd.center_part.py ${unit1} ${name} ${layer} ${xy} ${seed} ${neuron} ${filter_shape} ${unit2}
