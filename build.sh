cd ./Rogue
./build.sh
cd /public
rm -r ./francesco_sovrano
mkdir francesco_sovrano
chmod 700 francesco_sovrano
cd ./francesco_sovrano
virtualenv -p python3 .env
. .env/bin/activate
pip install --upgrade pip
pip install tensorflow matplotlib scipy scikit-image pyte