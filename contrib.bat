@ECHO OFF
git init
git add .
git commit -am "First try"
git remote add azure https://drugdisc.scm.azurewebsites.net:443/drugdisc.git
git push azure master
PAUSE