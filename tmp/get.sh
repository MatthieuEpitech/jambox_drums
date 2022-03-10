pid="$1"
file="$2"

for VARIABLE in drumified_beat_0_groovae_2bar_add_closed_hh.mid merge_groovae_2bar_add_closed_hh.mid drumified_beat_0_groovae_2bar_humanize.mid merge_groovae_2bar_humanize.mid drumified_beat_0_groovae_2bar_tap_fixed_velocity.mid merge_groovae_2bar_tap_fixed_velocity.mid drumified_beat_0_groovae_4bar.mid merge_groovae_4bar.mid merge_${file}_groovae_2bar_add_closed_hh.mid merge_${file}_groovae_2bar_humanize.mid merge_${file}_groovae_2bar_tap_fixed_velocity.mid merge_${file}_groovae_4bar.mid
do
	sudo docker cp $pid:$VARIABLE ./
done
