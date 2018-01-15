chmod -R 755 ./
cd ./rogue5.4.4-ant-r1.1.4
./configure
make clean
make -e EXTRA=-DSPAWN_MONSTERS
make install
