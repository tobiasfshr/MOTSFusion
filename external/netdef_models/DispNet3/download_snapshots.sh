#!/bin/bash
URL_BASE="https://lmb.informatik.uni-freiburg.de/resources/binaries/net_models/DispNet3"

download () {
	net=$1
	evo=$2
	state=$3
	subpath="$net/training/$evo/checkpoints"
	wget  "$URL_BASE/$subpath/snapshot-$state.data-00000-of-00001" -P $subpath
	wget  "$URL_BASE/$subpath/snapshot-$state.index" -P $subpath
	wget  "$URL_BASE/$subpath/snapshot-$state.meta" -P $subpath
}

download css 00__flyingThings3D.train__S_short 600000
download CSS 00__flyingThings3D.train__S_short 600000
download CSS-ft-kitti 00__kitti.train__S_custom 200000
