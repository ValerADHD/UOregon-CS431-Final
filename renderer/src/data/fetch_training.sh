wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip && unzip tandt_db.zip
mv db/* . && mv tandt/* .
rm -rf db tandt tandt_db.zip

