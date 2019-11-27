# Clone submodules:
git submodule init; git submodule update

# Download AI Physicist datasets:
wget https://space.mit.edu/home/tegmark/aiphysicist.tar.gz
tar xvzf aiphysicist.tar.gz -C datasets/
mv datasets/DATA/* datasets/
rm -r datasets/DATA
rm aiphysicist.tar.gz
