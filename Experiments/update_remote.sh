#Tunnels into cluster and pulls latest changges
ssh -tt -AY2C bstorpma@dnat.simula.no -p 60441 "ls; cd Master;
 git pull; exit"
