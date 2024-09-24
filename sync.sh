while ! sshpass -p 'xxxx' \
rsync --partial --append-verify --progress   -a -e 'ssh -p 22' /run/user/1000/gvfs/smb-share:server=172.21.251.64,share=nas/datasets_processed/Natural/nako_jpg /media/data/robert/datasets/; \
do sleep 5;done

while ! sshpass -p 'xxxx' \
rsync --partial --append-verify --progress   -a -e 'ssh -p 22' /media/data/robert/datasets/nako_jpg robert@141.39.166.79:/media/data/robert/datasets; 
do sleep 5;done