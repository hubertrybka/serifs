#!/bin/bash
cd data
mkdir train_data

cd train_data
wget -O big_dataset_std.parquet wget -O big_dataset_std.parquet https://www.dropbox.com/scl/fi/pdwlyza3l7i56fejl34vq/big_dataset_std.parquet?rlkey=g13z4witdik65x9d4lfzlclpq&dl=1
wget -O big_dataset_ECFP.paruet https://www.dropbox.com/scl/fi/l24c0qd2tfbjn2p9sukdy/big_dataset_ECFP.parquet?rlkey=vjxtauvbtxasiaxqvjhgzafzf&dl=1
wget -O train_morgan_512bits.parquet https://www.dropbox.com/s/bj37jlql9j1lcrr/train_morgan_512bits.parquet?dl=1

cd -
mkdir activity_data
cd activity_data
wget -O d2_klek_100nM.parquet https://www.dropbox.com/scl/fi/k96gdlxle87b3qak2f3g8/d2_klek_100nM_std.parquet?rlkey=etdw303bko452l3nilxtqa1gn&dl=1
wget -O d2_ECFP_100nM.parquet https://www.dropbox.com/scl/fi/qt1znjo79tffrdp0tx4dy/d2_ECFP_100nM_std.parquet?rlkey=dg3ie6zaw9k3lun1cefyk3597&dl=1
echo All datasets downloaded successfully
