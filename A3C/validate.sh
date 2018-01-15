pkill -9 -f python
pkill -9 -f rogue
rm -r log
mkdir log
cd ./log
mkdir screenshots
mkdir performance
cd ..
python3 ./lab/validate.py