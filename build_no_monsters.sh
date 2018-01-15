cd ./Rogue
./build_with_no_monsters.sh
cd ..
if [ ! -d "log" ]; then
	mkdir log
fi
chmod 700 log
cd ./log
virtualenv -p python3 .env
. .env/bin/activate
pip install --upgrade pip
pip install tensorflow matplotlib scipy scikit-image pyte