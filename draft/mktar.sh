ver=1
mkdir -p $ver
cp aastex61.cls $ver
cp *$ver.tex $ver
cp *.eps $ver
cp *.png $ver
#cp apjfonts.sty $ver
#rm $ver/Draft*.pdf
tar -cvf $ver.tar.gz $ver

