#Tunnels into cluster, pulls latest changes, adds and commits any new data, then exits.
ssh -AY2C bstorpma@dnat.simula.no -p 60441 "ls; rlogin g001; cd Master;
 git pull; exit"
