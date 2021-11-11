#Tunnels into cluster, pulls latest changes, adds and commits any new data, then exits.
ssh -tt -AY2C bstorpma@dnat.simula.no -p 60441 "ls; cd Master; ls;
 git pull; ls; exit"
