#!/bin/sh

echo 'Please uncomment the files you want to download ... '

################
## all test data - data_test.tar.gz
## includes full videos, features, transcripts, and groundtruth
## org link: https://drive.google.com/open?id=1obIuuHuV2vyk76dEJ2skLcj0iC5KJUcR
################
#FILE_ID="1obIuuHuV2vyk76dEJ2skLcj0iC5KJUcR"
#wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1obIuuHuV2vyk76dEJ2skLcj0iC5KJUcR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1obIuuHuV2vyk76dEJ2skLcj0iC5KJUcR" -O data_test_challenge.tar.gz && rm -rf ./cookies.txt
#wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1obIuuHuV2vyk76dEJ2skLcj0iC5KJUcR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1obIuuHuV2vyk76dEJ2skLcj0iC5KJUcR" -O data_test_challenge.tar.gz && rm -rf ./cookies.txt
#tar xzf data_test_challenge.tar.gz

echo 'DONE!'
