#!/bin/bash
URL_BASE="https://lmb.informatik.uni-freiburg.de/resources/binaries/net_models/Flownet3"

download () {
	net=$1
	evo=$2
	state=$3
	subpath="$net/training/$evo/checkpoints"
	wget "$URL_BASE/$subpath/snapshot-$state.data-00000-of-00001" -P $subpath
	wget "$URL_BASE/$subpath/snapshot-$state.index" -P $subpath
	wget "$URL_BASE/$subpath/snapshot-$state.meta" -P $subpath
}

download css 00__flyingThings3D.train__S_fine_half 250000
download CSS 00__flyingThings3D.train__S_fine_half 250000
download CSSR-ft-sd 00__SDMixture_no_hom__S_refinement 175000
download CSS-ft-kitti 00__kitti.train__S_custom 200000
download CSSR-ft-sintel 00__sintel_mixture__S_custom 25000
