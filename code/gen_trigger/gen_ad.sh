layer=fc6
xy=0
seed=1
unit1=81
unit2=694
neuron=1
size=0

name="${layer}_${seed}_${unit1}_${unit2}_${neuron}_${size}"
echo ${unit1} ${name} ${layer} ${xy} ${seed} ${neuron} ${size} ${unit2}
python ./act_max.tvd.center_part.py ${unit1} ${name} ${layer} ${xy} ${seed} ${neuron} ${size} ${unit2}
