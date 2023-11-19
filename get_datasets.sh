#!/bin/bash
cd data
wget -O RNN_dataset_KRFP.parquet https://www.dropbox.com/scl/fi/zwwnzfu7igkogy73wofmd/RNN_dataset_KRFP.parquet?rlkey=3sy01u7hdgwup10glw4cxqduc&dl=1
wget -O RNN_dataset_ECFP.parquet https://www.dropbox.com/scl/fi/gfqlz55bo1ql4t9mp9648/RNN_dataset_ECFP.parquet?rlkey=g7j5vsvc9cea9genassod9wni&dl=1
wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/misnnxgj31ce21qeczfo4/d2_klek_100nM_std.parquet?rlkey=ueslf82ev4014vn7eqkzql4c5&dl=1
wget -O d2_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/eo21hwgbboahy11xb5bt1/d2_ECFP_100nM_std.parquet?rlkey=mnbcfs28r94mhu86pxq5x1a1c&dl=1
cd ../
cd models
mkdir GRUv3_KRFP
cd GRUv3_KRFP
wget -O hyperparameters.ini https://www.dropbox.com/scl/fi/gcia3vh4k5y2qpvwzgc96/hyperparameters.ini?rlkey=1xvjenbsu01wo9vtv6r0qup0o&dl=1
wget -O epoch_200.pt https://www.dropbox.com/scl/fi/o2lryveji4tbxmgmfgl6e/epoch_200.pt?rlkey=dnl45pu0dswicgxdu2art76yi&dl=1
cd ../
mkdir GRUv3_ECFP
cd GRUv3_ECFP
wget -O epoch_150.pt https://www.dropbox.com/scl/fi/n98tkbvwmnihv2hgszbf2/epoch_150.pt?rlkey=iuy4ncyt5vhm3t2mc2judt6iz&dl=1
wget -O hyperparameters.ini https://www.dropbox.com/scl/fi/ks0qwc6cj03da27xuvo4v/hyperparameters.ini?rlkey=a2d7y2xa8uqgdny0wk18va4ct&dl=1
cd ../
mkdir SVC_klek_200
cd SVC_klek_200
wget -O model.pkl https://www.dropbox.com/scl/fi/362e8ppafrkvx8dlov5ik/d2.pkl?rlkey=kku0rlyd05wdrltv11oveq00o&dl=1
cd ../
mkdir SVC_ECFP_150
cd SVC_ECFP_150
wget -O model.pkl https://www.dropbox.com/scl/fi/1mmjceoisvhis9j0t16sq/d2.pkl?rlkey=madnlty9hl28qnx0rm79whoru&dl=1
echo All datasets and models downloaded successfully