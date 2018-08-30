# Download the prebuilt models

# from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805#gistcomment-2359248
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}


pushd

gdrive_download 1pDagzQup--DPOrb21-dIPydwInTSHkMU fn1.7-pretrained-targetid.tar.gz
gdrive_download 1K6Nc9d4yRai7a1YUSq3EI2-2rivm2uOi fn1.7-pretrained-frameid.tar.gz
gdrive_download 1aBQH6gKx-50xcKUgoqPGsRhVgc4THYgs fn1.7-pretrained-argid.tar.gz

tar -xzf fn1.7-pretrained-targetid.tar.gz
tar -xzf fn1.7-pretrained-frameid.tar.gz
tar -xzf fn1.7-pretrained-argid.tar.gz

rm fn1.7-pretrained-*

popd