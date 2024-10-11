while read fname; do
	if test -f "./${fname}.xml"; then
		echo "./${fname}.xml"
	else
		touch "./${fname}.xml"
	fi
done < list.txt
