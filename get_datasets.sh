#!/bin/bash
cd data
wget -O RNN_dataset_KRFP.parquet https://www.dropbox.com/scl/fi/zwwnzfu7igkogy73wofmd/RNN_dataset_KRFP.parquet?rlkey=3sy01u7hdgwup10glw4cxqduc&dl=1
wget -O RNN_dataset_ECFP.parquet https://www.dropbox.com/scl/fi/gfqlz55bo1ql4t9mp9648/RNN_dataset_ECFP.parquet?rlkey=g7j5vsvc9cea9genassod9wni&dl=1
wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/misnnxgj31ce21qeczfo4/d2_klek_100nM_std.parquet?rlkey=ueslf82ev4014vn7eqkzql4c5&dl=1
wget -O d2_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/eo21hwgbboahy11xb5bt1/d2_ECFP_100nM_std.parquet?rlkey=mnbcfs28r94mhu86pxq5x1a1c&dl=1
cd -
cd models
wget -O GRUv3_ECFP https://www.dropbox.com/scl/fo/ngcc4zpav4gop6ck77ps3/h?rlkey=3kxszlhmqoxe3yjcxn657iynp&dl=1
wget -O GRUv3_KRFP https://www.dropbox.com/scl/fo/10wz3f7titnvjuuwphmjz/h?rlkey=a966jh4vew21t41y8ao9xsywp&dl=1