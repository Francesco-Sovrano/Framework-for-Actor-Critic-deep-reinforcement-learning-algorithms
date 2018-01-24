# Bazel
apt-get install software-properties-common swig
add-apt-repository ppa:webupd8team/java
apt-get update
apt-get install oracle-java8-installer
echo "deb http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | apt-key add -
apt-get update
apt-get install bazel
# DeepMind lab dependencies
apt install lua5.1 liblua5.1-0-dev libffi-dev gettext freeglut3-dev libsdl2-dev libosmesa6-dev python-dev python-numpy realpath
# Build DeepMind lab
git clone https://github.com/deepmind/lab.git
cd lab
# Build the Python interface to DeepMind Lab with software rendering
bazel build :deepmind_lab.so --define headless=osmesa
# Install UNREAL
cd lab
git clone https://github.com/miyosuda/unreal.git